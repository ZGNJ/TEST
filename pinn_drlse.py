"""PINN for the user-supplied DRLSE. All spatial units are original pixels.
Independent per-image PINNs with image-only Otsu initialization and terminal GT loss by default.
"""
import argparse, csv, hashlib, json, math, random, time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parent

def threshold_initial_mask(gray, polarity="auto"):
    """Otsu initialization from the image only; no label or semantic priors."""
    if polarity not in ("auto", "dark", "bright"):
        raise ValueError(f"Unknown initialization polarity: {polarity}")
    if gray.min() == gray.max():
        raise ValueError("Cannot initialize a contour by thresholding a constant image")
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = gray <= threshold
    border = np.zeros(gray.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    dark_border_fraction = float(dark[border].mean())
    selected = polarity
    if polarity == "auto":
        # The class occupying less of the border is the foreground heuristic.
        # Break ties by total area, then choose dark deterministically.
        dark_score = (int(dark[border].sum()), int(dark.sum()))
        bright_score = (int((~dark)[border].sum()), int((~dark).sum()))
        selected = "dark" if dark_score <= bright_score else "bright"
    mask = dark if selected == "dark" else ~dark
    if not mask.any() or mask.all():
        raise ValueError("Otsu initialization requires both foreground and background")
    return mask, dict(threshold_method="otsu", otsu_threshold=float(threshold),
                      requested_polarity=polarity, selected_polarity=selected,
                      dark_border_fraction=dark_border_fraction)

def prepare_image(origin, out, sigma, init_polarity="auto"):
    """Generic grayscale smoothing and thresholding, without access to GT."""
    rgb = np.asarray(Image.open(origin).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    # Use one Gaussian scale for both the threshold image and image gradients.
    smoothed = ndi.gaussian_filter(gray, sigma)
    init_gray = np.clip(np.rint(smoothed), 0, 255).astype(np.uint8)
    mask, threshold_metadata = threshold_initial_mask(init_gray, init_polarity)
    # Preserve all threshold components and holes; do not impose lesion geometry.
    sdf = ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)
    # Preserve 0..255 intensity when computing the supplied edge indicator.
    ix = ndi.gaussian_filter(gray, sigma, order=(0, 1))
    iy = ndi.gaussian_filter(gray, sigma, order=(1, 0))
    g = 1 / (1 + ix * ix + iy * iy)
    gy, gx = np.gradient(g)
    for name, array in [("preprocessed.png", init_gray),
                        ("initial_mask.png", mask.astype(np.uint8) * 255),
                        ("edge_indicator.png", (g * 255).astype(np.uint8))]:
        Image.fromarray(array).save(out / name)
    np.save(out / "initial_sdf.npy", sdf.astype(np.float32))
    np.save(out / "edge_indicator.npy", g.astype(np.float32))
    h, w = mask.shape
    metadata = dict(width=w, height=h, **threshold_metadata,
                    preprocessing="grayscale_gaussian", initialization_source="original_image_only",
                    gt_used_for_initialization=False, initial_foreground_pixels=int(mask.sum()),
                    sigma_original_pixels=sigma, intensity_range_for_g="0..255",
                    initialization_feature="smoothed_grayscale")
    return rgb, sdf.astype(np.float32), g.astype(np.float32), gx.astype(np.float32), gy.astype(np.float32), metadata

def sample_array(array, xy, w, h):
    return ndi.map_coordinates(array, [(xy[:, 1] + 1) * (h - 1) / 2,
                                       (xy[:, 0] + 1) * (w - 1) / 2],
                               order=1, mode="nearest").astype(np.float32)

def sample_xy(rng, count, field, fraction=0.7, band_width=8.0):
    h, w = field.shape
    n = int(count * fraction)
    locations = np.argwhere(np.abs(field) < band_width)
    if not len(locations):
        raise ValueError("Zero-level contour disappeared")
    pixels = locations[rng.integers(0, len(locations), n)][:, [1, 0]].astype(np.float32)
    pixels += rng.uniform(-.5, .5, pixels.shape).astype(np.float32)
    uniform = rng.uniform([0, 0], [w-1, h-1], (count-n, 2)).astype(np.float32)
    pixels = np.concatenate([pixels, uniform])
    pixels[:, 0] = pixels[:, 0].clip(0, w-1)
    pixels[:, 1] = pixels[:, 1].clip(0, h-1)
    return pixels / np.array([(w-1)/2, (h-1)/2], np.float32) - 1

class SmoothNet(nn.Module):
    """Smooth Fourier-feature MLP, differentiated twice with autograd."""
    def __init__(self, temporal=False, width=128, depth=4, epsilon=1.5, boundary="free"):
        super().__init__()
        self.temporal = temporal
        self.epsilon = epsilon
        if boundary not in ("free", "fixed"):
            raise ValueError(f"Unknown spatial boundary mode: {boundary}")
        self.boundary = boundary
        self.register_buffer("freq", torch.tensor([1, 2, 4, 8, 16], dtype=torch.float32) * math.pi)
        layers = [nn.Linear(22 + 4*int(temporal), width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2 if temporal else 1)]
        self.layers = nn.Sequential(*layers)
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        if temporal:
            nn.init.zeros_(self.layers[-1].weight)
    def forward(self, coords, base_phi=None):
        xy = coords[:, :2]
        phase = xy[:, :, None] * self.freq[None, None, :]
        features = [xy, torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)]
        if self.temporal:
            if base_phi is None:
                raise ValueError("Temporal PINN requires its differentiable initial field")
            q = base_phi / self.epsilon
            envelope = torch.exp(-0.5*q*q)
            features += [coords[:, 2:3], torch.tanh(q), envelope, q*envelope]
            heads = self.layers(torch.cat(features, dim=1))
            velocity = heads[:, 0:1]+envelope*heads[:, 1:2]
            if self.boundary == "fixed":
                # Optional legacy boundary: freezes even incorrect initial border signs.
                wall = (1-xy[:, 0:1]**2)*(1-xy[:, 1:2]**2)
                velocity = wall*velocity
            # Free spatial borders can be corrected by the terminal loss. The hard
            # initial-time condition is still enforced by phi=base+dt*tau*velocity.
            return velocity
        return self.layers(torch.cat(features, dim=1))

class EvolvedField(nn.Module):
    def __init__(self, initial, distance_scale):
        super().__init__()
        self.initial, self.distance_scale = initial, distance_scale
        self.steps, self.dts = nn.ModuleList(), []
    def forward(self, xy):
        phi = self.distance_scale * self.initial(xy)
        if self.steps:
            z = torch.cat([xy, torch.ones_like(xy[:, :1])], dim=1)
            for dt, step in zip(self.dts, self.steps):
                phi = phi + dt * step(z, phi)
        return phi

def grad(value, coords, create_graph=True):
    return torch.autograd.grad(value, coords, torch.ones_like(value),
                               create_graph=create_graph, retain_graph=True)[0]

def spatial_derivatives(value, coords, sx, sy, create_graph=True):
    first = grad(value, coords)
    dx, dy = first[:, 0:1]*sx, first[:, 1:2]*sy
    second_x, second_y = grad(dx, coords, create_graph), grad(dy, coords, create_graph)
    return value, dx, dy, second_x[:, 0:1]*sx, second_x[:, 1:2]*sy, second_y[:, 1:2]*sy

def double_well(s):
    dp = torch.where(s <= 1, torch.sinc(2*s), 1-1/s)
    a = 2*math.pi
    taylor = -a*a/3 + a**4*s*s/30 - a**6*s**4/840
    stable = s.clamp_min(.02)
    exact = (a*stable*torch.cos(a*stable)-torch.sin(a*stable))/(a*stable**3)
    ratio = torch.where(s <= 1, torch.where(s < .02, taylor, exact), 1/s**3)
    return dp, ratio

def drlse_rhs(phi, px, py, pxx, pxy, pyy, g, gx, gy,
              mu=.2, lam=5., alpha=-.5, epsilon=1.5, eta=1e-6):
    s = torch.sqrt(px*px+py*py+eta)
    lap = pxx+pyy
    directional = px*px*pxx+2*px*py*pxy+py*py*pyy
    dp, ratio = double_well(s)
    regularization = dp*lap+ratio*directional
    curvature = lap/s-directional/s**3
    delta = torch.where(phi.abs() <= epsilon, (1+torch.cos(math.pi*phi/epsilon))/(2*epsilon),
                        torch.zeros_like(phi))
    edge = delta*(gx*px/s+gy*py/s+g*curvature)
    return mu*regularization+lam*edge+alpha*g*delta

@torch.no_grad()
def rasterize(field, w, h, device, batch=16384):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xy = np.stack([xx.ravel()*2/(w-1)-1, yy.ravel()*2/(h-1)-1], axis=1)
    return np.concatenate([field(torch.from_numpy(xy[i:i+batch]).to(device)).cpu().numpy()
                           for i in range(0, len(xy), batch)]).reshape(h, w)

def pretrain_initial(sdf, out, args, device):
    h, w = sdf.shape
    net = SmoothNet(width=args.width).to(device)
    scale, rng = max(w, h)/4, np.random.default_rng(args.seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    xy = sample_xy(rng, 150000, sdf, .6, 12)
    x = torch.from_numpy(xy).to(device)
    y = torch.from_numpy(sample_array(sdf, xy, w, h)[:, None]).to(device)
    history, started = [], time.perf_counter()
    for step in range(1, args.initial_steps+1):
        ids = torch.randint(len(x), (args.batch,), device=device)
        prediction = scale*net(x[ids])
        weights = 1+5*torch.exp(-y[ids].abs()/12)
        loss = (weights*((prediction-y[ids])/scale)**2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (int(args.initial_steps*.7), int(args.initial_steps*.9)):
            for group in optimizer.param_groups:
                group["lr"] *= .25
        if step % 250 == 0 or step == 1 or step == args.initial_steps:
            band = y[ids].abs() < 5
            row = dict(stage="initial", step=step, loss=loss.item(),
                       band_rmse_px=((prediction[band]-y[ids][band])**2).mean().sqrt().item())
            history.append(row)
            print(json.dumps(row), flush=True)
    field = EvolvedField(net, scale).to(device)
    for parameter in field.parameters():
        parameter.requires_grad_(False)
    fitted = rasterize(field, w, h, device)
    band = np.abs(sdf) < 5
    audit = dict(global_rmse_px=float(np.sqrt(np.mean((fitted-sdf)**2))),
                 narrow_band_rmse_px=float(np.sqrt(np.mean((fitted[band]-sdf[band])**2))),
                 sign_agreement=float(((fitted < 0) == (sdf < 0)).mean()),
                 seconds=time.perf_counter()-started)
    np.save(out/"neural_initial_phi.npy", fitted)
    Image.fromarray((fitted < 0).astype(np.uint8)*255).save(out/"neural_initial_mask.png")
    print("INITIAL_AUDIT "+json.dumps(audit), flush=True)
    return field, fitted, audit, history

def derivative_cache(field, xy, device, w, h, batch=2048):
    blocks = []
    for i in range(0, len(xy), batch):
        coords = torch.from_numpy(xy[i:i+batch]).to(device).requires_grad_(True)
        vals = spatial_derivatives(field(coords), coords, 2/(w-1), 2/(h-1), False)
        blocks.append(torch.cat(vals, dim=1).detach())
    return torch.cat(blocks)

def make_pool(field, current, g, gx, gy, count, rng, device):
    h, w = current.shape
    xy = sample_xy(rng, count, current)
    base = derivative_cache(field, xy, device, w, h)
    coefficients = np.stack([sample_array(a, xy, w, h) for a in (g, gx, gy)], axis=1)
    return torch.from_numpy(xy).to(device), base, torch.from_numpy(coefficients).to(device)

def window_residual(net, coords, base, coeffs, dt, w, h, args, create_graph=True):
    # Exact local second-order jet of the frozen neural initial field.
    # The offset is zero in value, but preserves its cached first/second derivatives.
    offset = coords[:, :2]-coords[:, :2].detach()
    dx, dy = offset[:, 0:1]*(w-1)/2, offset[:, 1:2]*(h-1)/2
    base_jet = (base[:, 0:1]+base[:, 1:2]*dx+base[:, 2:3]*dy
                +0.5*base[:, 3:4]*dx*dx+base[:, 4:5]*dx*dy+0.5*base[:, 5:6]*dy*dy)
    v = net(coords, base_jet)
    derivs = spatial_derivatives(v, coords, 2/(w-1), 2/(h-1), create_graph)
    temporal, tau = grad(v, coords, create_graph)[:, 2:3], coords[:, 2:3]
    phi_derivs = [base[:, k:k+1]+dt*tau*val for k, val in enumerate(derivs)]
    phi_t = v+tau*temporal
    rhs = drlse_rhs(*phi_derivs, *(coeffs[:, k:k+1] for k in range(3)),
                    args.mu, args.lam, args.alpha, args.epsilon, args.eta)
    return phi_t-rhs, phi_t, rhs, phi_derivs[0]

def independent_audit(net, field, current, g, gx, gy, args, device, seed):
    h, w = current.shape
    rng, audits = np.random.default_rng(seed), {}
    for name, fraction in [("narrow_band", 1.), ("uniform", 0.)]:
        xy = sample_xy(rng, 6000, current, fraction, 5)
        base = derivative_cache(field, xy, device, w, h)
        coeffs = torch.from_numpy(np.stack([sample_array(a, xy, w, h) for a in (g, gx, gy)], axis=1)).to(device)
        tau, values = rng.uniform(0, 1, (len(xy), 1)).astype(np.float32), []
        for i in range(0, len(xy), 1000):
            coords = torch.from_numpy(np.concatenate([xy[i:i+1000], tau[i:i+1000]], axis=1)).to(device).requires_grad_(True)
            residual, temporal, rhs, _ = window_residual(net, coords, base[i:i+1000],
                                                        coeffs[i:i+1000], args.dt, w, h, args, False)
            frozen = drlse_rhs(*(base[i:i+1000, k:k+1] for k in range(6)),
                               *(coeffs[i:i+1000, k:k+1] for k in range(3)),
                               args.mu, args.lam, args.alpha, args.epsilon, args.eta)
            values.append(torch.cat([residual, temporal, rhs, frozen], dim=1).detach().cpu().numpy())
        result = np.concatenate(values)
        rms = np.sqrt(np.mean(result**2, axis=0))
        audits[name] = dict(residual_rms=float(rms[0]), residual_abs_p95=float(np.quantile(np.abs(result[:, 0]), .95)),
                            phi_t_rms=float(rms[1]), rhs_rms=float(rms[2]),
                            relative_residual=float(rms[0]/(rms[1]+rms[2]+1e-8)),
                            frozen_initial_residual_rms=float(rms[3]),
                            residual_reduction_vs_frozen=float(1-rms[0]/(rms[3]+1e-8)))
    return audits

def save_checkpoint(field, path, args, w, h):
    torch.save(dict(state_dict=field.state_dict(), distance_scale=field.distance_scale, dts=field.dts,
                    width=args.width, image_width=w, image_height=h, arguments=vars(args)), path)

def train_windows(field, initial_phi, g, gx, gy, out, args, device, supervision=None):
    h, w = initial_phi.shape
    rng, current, history, audits = np.random.default_rng(args.seed+10), initial_phi, [], []
    supervision_rng = np.random.default_rng(args.seed+20)
    supervision_generator = torch.Generator(device=device).manual_seed(args.seed+20)
    for window in range(args.windows):
        started = time.perf_counter()
        net = SmoothNet(temporal=True, width=args.width, epsilon=args.epsilon,
                        boundary=args.boundary).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        xy, base, coeffs = make_pool(field, current, g, gx, gy, args.pool, rng, device)
        gt_pool = (make_supervision_pool(field, supervision, supervision_rng, args.pool, device)
                   if supervision is not None and window == args.windows-1 and args.gt_weight > 0 else None)
        for step in range(1, args.window_steps+1):
            ids = torch.randint(len(xy), (args.batch,), device=device)
            coords = torch.cat([xy[ids], torch.rand((args.batch, 1), device=device)], dim=1).requires_grad_(True)
            residual, phi_t, rhs, phi = window_residual(net, coords, base[ids], coeffs[ids], args.dt, w, h, args)
            weights = 1+8*(phi.detach().abs() < 3).to(phi.dtype)
            loss_physics = (weights*residual**2).mean()
            loss_bc = torch.zeros((), device=device)
            loss_gt = torch.zeros((), device=device)
            gt_terms = {"bce": loss_gt, "sdf": loss_gt, "dice": loss_gt}
            if gt_pool is not None:
                gt_xy, gt_base, gt_labels, gt_sdf = gt_pool
                gt_ids = torch.randint(len(gt_xy), (args.batch,), device=device,
                                       generator=supervision_generator)
                gt_coords = torch.cat([gt_xy[gt_ids], torch.ones((args.batch, 1), device=device)], dim=1)
                terminal_phi = gt_base[gt_ids]+args.dt*net(gt_coords, gt_base[gt_ids])
                loss_gt, gt_terms = terminal_supervised_loss(
                    terminal_phi, gt_labels[gt_ids], gt_sdf[gt_ids], args.epsilon,
                    args.gt_sdf_weight, args.gt_distance_scale, args.gt_dice_weight)
            loss = loss_physics+args.gt_weight*loss_gt
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            if step in (int(args.window_steps*.7), int(args.window_steps*.9)):
                for group in optimizer.param_groups:
                    group["lr"] *= .3
            if step % 250 == 0 or step == 1 or step == args.window_steps:
                row = dict(stage="physics", window=window+1, step=step, loss=loss.item(),
                           pde_loss=loss_physics.item(), bc_loss=loss_bc.item(),
                           supervised_loss=loss_gt.item(), gt_bce=gt_terms["bce"].item(),
                           gt_sdf=gt_terms["sdf"].item(), gt_dice=gt_terms["dice"].item(),
                           residual_rms=residual.detach().square().mean().sqrt().item(),
                           phi_t_rms=phi_t.detach().square().mean().sqrt().item())
                history.append(row)
                print(json.dumps(row), flush=True)
        net.eval()
        audit = independent_audit(net, field, current, g, gx, gy, args, device, args.seed+1000+window)
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        field.steps.append(net)
        field.dts.append(args.dt)
        previous, current = current, rasterize(field, w, h, device)
        audit.update(window=window+1, t_start=window*args.dt, t_end=(window+1)*args.dt,
                     terminal_gt_supervised=gt_pool is not None,
                     initial_condition_error=0., initial_condition_enforcement="hard: phi=base+dt*tau*v",
                     foreground_before=int((previous < 0).sum()), foreground_after=int((current < 0).sum()),
                     changed_sign_pixels=int(((previous < 0) != (current < 0)).sum()),
                     seconds=time.perf_counter()-started)
        audits.append(audit)
        np.save(out/f"phi_window_{window+1:02d}.npy", current)
        Image.fromarray((current < 0).astype(np.uint8)*255).save(out/f"mask_window_{window+1:02d}.png")
        (out/"physics_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
        save_checkpoint(field, out/"model.pt", args, w, h)
        print("WINDOW_AUDIT "+json.dumps(audit), flush=True)
    return current, history, audits

def mask_topology(mask):
    """Count 4-connected foreground and complementary 8-connected background holes."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("Foreground masks must be nonempty two-dimensional arrays")
    foreground_structure = ndi.generate_binary_structure(2, 1)
    background_structure = ndi.generate_binary_structure(2, 2)
    _, components = ndi.label(mask, structure=foreground_structure)
    filled = ndi.binary_fill_holes(mask, structure=background_structure)
    _, holes = ndi.label(filled & ~mask, structure=background_structure)
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    border_pixels = int((mask & border).sum())
    return dict(components=int(components), holes=int(holes),
                foreground_pixels=int(mask.sum()), border_foreground_pixels=border_pixels,
                touches_border=border_pixels > 0,
                foreground_connectivity=4, background_connectivity=8)


def finalize_foreground(mask, mode="single"):
    """Select one connected prediction without using image labels or altering initialization."""
    if mode not in ("single", "all"):
        raise ValueError(f"Unknown foreground mode: {mode}")
    raw = np.asarray(mask, dtype=bool)
    before = mask_topology(raw)
    final = raw.copy()
    removed, filled_count = 0, 0
    if mode == "single" and raw.any():
        labels, count = ndi.label(raw, structure=ndi.generate_binary_structure(2, 1))
        sizes = np.bincount(labels.ravel(), minlength=count+1)
        sizes[0] = 0
        # np.argmax breaks equal-area ties by deterministic raster scan label order.
        final = labels == int(np.argmax(sizes))
        removed = int((raw & ~final).sum())
        filled = ndi.binary_fill_holes(final, structure=ndi.generate_binary_structure(2, 2))
        filled_count = int((filled & ~final).sum())
        final = filled
    after = mask_topology(final)
    status = ("no_foreground" if not final.any() else
              "full_image_foreground" if final.all() else "foreground_detected")
    audit = dict(mode=mode, gt_used=False,
                 method="largest_4_connected_component_then_fill_8_connected_background_holes"
                 if mode == "single" else "identity",
                 raw_topology=before, final_topology=after,
                 removed_foreground_pixels=removed, filled_hole_pixels=filled_count,
                 changed_pixels=int((raw != final).sum()), status=status,
                 border_policy="preserve; do not crop or force background on image border")
    return final, audit


def foreground_sdf(mask):
    """Negative-inside distance field of the exported mask, separate from neural phi."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, max(mask.shape), dtype=np.float32)
    if mask.all():
        # The outside of the image is background for a fully foreground image.
        inside = ndi.distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]
        return -inside.astype(np.float32)
    return (ndi.distance_transform_edt(~mask)-ndi.distance_transform_edt(mask)).astype(np.float32)


def mask_contours(mask):
    """Closed pixel contours, including image-border segments, with no border deletion."""
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, constant_values=0)
    contours, _ = cv2.findContours(padded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [contour-1 for contour in contours]


def contour_distances(a, b):
    ea, eb = a ^ ndi.binary_erosion(a), b ^ ndi.binary_erosion(b)
    if not ea.any() or not eb.any():
        return dict(hausdorff95_px=None, mean_symmetric_boundary_distance_px=None)
    da, db = ndi.distance_transform_edt(~ea)[eb], ndi.distance_transform_edt(~eb)[ea]
    return dict(hausdorff95_px=float(np.quantile(np.concatenate([da, db]), .95)),
                mean_symmetric_boundary_distance_px=float((da.sum()+db.sum())/(len(da)+len(db))))

def metrics(pred, gt):
    tp, fp, fn, tn = (int(a.sum()) for a in (pred & gt, pred & ~gt, ~pred & gt, ~pred & ~gt))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, dice=2*tp/max(2*tp+fp+fn, 1), iou=tp/max(tp+fp+fn, 1),
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1), accuracy=(tp+tn)/pred.size,
                foreground_pixels=int(pred.sum()), gt_foreground_pixels=int(gt.sum()), **contour_distances(pred, gt))

def evaluate_and_export(rgb, phi, initial_phi, gt_path, out, config, audit, history):
    # Evaluation reuses the same GT; terminal-supervised runs already used it in training.
    gt_raw = np.asarray(Image.open(gt_path).convert("L"))
    if gt_raw.shape != phi.shape:
        raise ValueError(f"GT {gt_raw.shape} differs from phi {phi.shape}")
    gt, raw_pred = gt_raw > 127, phi < 0
    pred, postprocessing = finalize_foreground(
        raw_pred, config["arguments"].get("foreground_mode", "single"))
    result = dict(metrics(pred, gt), **{k: v for k, v in postprocessing["final_topology"].items()
                                     if k != "foreground_pixels"})
    raw_result = dict(metrics(raw_pred, gt), **{k: v for k, v in postprocessing["raw_topology"].items()
                                             if k != "foreground_pixels"})
    initial_result = metrics(initial_phi < 0, gt)
    # Preserve the neural solution for checkpoints/PDE audits; never replace it by a cleaned SDF.
    np.save(out/"phi_final.npy", phi.astype(np.float32))
    np.save(out/"segmentation_sdf.npy", foreground_sdf(pred))
    Image.fromarray(raw_pred.astype(np.uint8)*255).save(out/"raw_segmentation_mask.png")
    Image.fromarray(pred.astype(np.uint8)*255).save(out/"segmentation_mask.png")
    extracted = rgb.copy()
    extracted[~pred] = 0
    Image.fromarray(extracted).save(out/"segmented_image.png")
    contours = mask_contours(pred)
    overlay = rgb.copy()
    cv2.drawContours(overlay, contours, -1, (255, 35, 30), 1)
    Image.fromarray(overlay).save(out/"contour_overlay.png")
    raw_overlay = rgb.copy()
    cv2.drawContours(raw_overlay, mask_contours(raw_pred), -1, (255, 35, 30), 1)
    Image.fromarray(raw_overlay).save(out/"raw_contour_overlay.png")
    comparison = rgb.copy()
    cv2.drawContours(comparison, mask_contours(gt), -1, (30, 245, 90), 1)
    cv2.drawContours(comparison, contours, -1, (255, 35, 30), 1)
    Image.fromarray(comparison).save(out/"prediction_vs_gt.png")
    errors = np.zeros_like(rgb)
    errors[pred & gt], errors[pred & ~gt], errors[~pred & gt] = (220, 220, 220), (255, 45, 45), (40, 120, 255)
    Image.fromarray(errors).save(out/"error_map.png")
    report = dict(final=result, raw_prediction=raw_result, neural_initial=initial_result,
                  postprocessing=postprocessing,
                  zero_level_set="raw_segmentation_mask=(phi_final<0); segmentation_mask=(segmentation_sdf<0)",
                  field_usage="phi_final/model/physics_audit describe the raw neural field; segmentation_sdf describes the exported final mask",
                  gt_usage=("Terminal supervision and training-fit evaluation; white foreground"
                            if config["gt_supervised"] else "Evaluation only; white foreground"),
                  metric_scope=config["metric_scope"],
                  pde=config, numerical_audit=audit, contour_change=contour_distances(initial_phi < 0, pred),
                  raw_contour_change=contour_distances(initial_phi < 0, raw_pred))
    (out/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out/"training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = sorted(set().union(*(row.keys() for row in history)))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    make_figure(rgb, pred, gt, phi, history, result, out, config["gt_supervised"])
    write_report(config, report, out)
    print("FINAL_METRICS "+json.dumps(result), flush=True)
    return report

def make_figure(rgb, pred, gt, phi, history, result, out, gt_supervised=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("Original image")
    axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("Ground truth (training target)" if gt_supervised else "Ground truth (evaluation only)")
    axes[0, 2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[0, 2].set_title("Final foreground mask")
    axes[1, 0].imshow(np.asarray(Image.open(out/"contour_overlay.png")))
    axes[1, 0].set_title("Final foreground contour")
    axes[1, 1].imshow(np.asarray(Image.open(out/"prediction_vs_gt.png")))
    axes[1, 1].set_title("Final foreground (red) / GT (green)")
    axes[1, 2].imshow(np.asarray(Image.open(out/"error_map.png"))); axes[1, 2].set_title("TP white / FP red / FN blue")
    fig.suptitle(f"DRLSE-PINN final foreground | Dice={result['dice']:.4f}, IoU={result['iou']:.4f}", fontsize=15, y=.99)
    fig.tight_layout(); fig.savefig(out/"result_summary.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    raw = phi < 0
    raw_metrics = metrics(raw, gt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mask, title in zip(axes, (raw, pred),
                              (f"Raw neural mask | Dice {raw_metrics['dice']:.4f}",
                               f"Final foreground | Dice {result['dice']:.4f}")):
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out/"raw_vs_final.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    physics = [r for r in history if r["stage"] == "physics"]
    for window in sorted({r["window"] for r in physics}):
        rows = [r for r in physics if r["window"] == window]
        ax.semilogy([r["step"] for r in rows], [r["pde_loss"] for r in rows], label=f"Window {window}")
    if gt_supervised:
        rows = [r for r in physics if r.get("supervised_loss", 0) > 0]
        ax.semilogy([r["step"] for r in rows], [r["supervised_loss"] for r in rows],
                    "--", label="Terminal GT loss (before weight)")
    ax.set(xlabel="Adam iteration", ylabel="Loss", title="PINN physics and supervision" if gt_supervised else "PINN physics")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/"training_loss.png", dpi=150); plt.close(fig)

def write_report(config, report, out):
    m, a, audits = report["final"], report["neural_initial"], report["numerical_audit"]
    raw, post = report["raw_prediction"], report["postprocessing"]
    lines = ["# DRLSE 的 PINN 图像分割结果", "",
             f"原图 {Path(config['arguments']['origin']).name}；标注 {Path(config['arguments']['gt']).name}。白色为前景。", "",
             ("GT 参与最终时刻的监督损失；此处指标是训练样本拟合指标，不是独立验证或测试性能。"
              if config["gt_supervised"] else "GT 仅用于评价，不参与训练。"), "",
             "负内正外：原始神经预测为 phi(x,y,T)<0。最终导出掩膜经过下述前景整理，与原始神经场分开保存和评价。", "",
             ("single 模式保留最大 4 连通前景并填充内部 8 连通背景孔洞；不做大尺度闭运算，不强制删除触边目标。"
              if post["mode"] == "single" else "all 模式保留全部原始预测像素，不做前景整理。"),
             "GT 不参与前景整理、分量选择或填洞。神经场、模型权重与 PDE 审计均保持原始预测；最终掩膜 SDF 单独保存。", "",
             f"前景分量数 {raw['components']} → {m['components']}；孔洞数 {raw['holes']} → {m['holes']}；"
             f"移除分量像素 {post['removed_foreground_pixels']}，填洞像素 {post['filled_hole_pixels']}。"
             f"最终触边像素 {m['border_foreground_pixels']}；状态 {post['status']}。", "",
             "## 指标", "", "| 指标 | 最终前景 | 原始神经预测 | 神经初始轮廓 |", "|---|---:|---:|---:|"]
    for key, title in [("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall"),
                       ("accuracy", "Accuracy"), ("hausdorff95_px", "HD95 / 原图像素"),
                       ("mean_symmetric_boundary_distance_px", "平均对称边界距离 / 原图像素")]:
        left = f"{m[key]:.6f}" if m[key] is not None else "N/A"
        middle = f"{raw[key]:.6f}" if raw[key] is not None else "N/A"
        right = f"{a[key]:.6f}" if a[key] is not None else "N/A"
        lines.append(f"| {title} | {left} | {middle} | {right} |")
    lines += ["", "## 方法", "",
              "完整使用双势阱 d_p、紧支撑 Dirac 与边缘项。phi 的时间导数及空间一、二阶导数均用自动微分，按原始像素换算；图像系数 g 及其梯度由图像网格计算并双线性采样，其中 g 梯度使用 numpy.gradient。", "",
              "原图转灰度并以 sigma 高斯平滑，再以 Otsu 阈值产生初始掩膜及有符号距离场。auto 前景极性选择图像边框占比更少的一类，同占比时选择总面积更少的一类；可显式指定 dark 或 bright。保留阈值结果的多连通域和孔洞。保留 0–255 灰度计算 g。GT 不参与预处理、阈值选择、初始掩膜或神经初始场拟合。", "",
              "默认开启最终时刻 GT 监督：在与 GT 内容无关的均匀像素网格上计算 BCE、Soft Dice 与窄带 SDF 损失，小图使用全网格，大图使用随机子集。GT 仅提供损失目标（训练结束后也用于评价）；物理采样使用独立随机流。多时间窗时只有最后一窗加入 GT 损失。", "",
              "初始神经网络拟合原图产生的有符号距离场。演化网络加入可微的初始 phi 窄带特征，第二阶局部导数缓存严格按链式法则传播。每窗 phi=phi_base+dt*tau*v_theta 硬约束初始条件；前一窗神经终点直接成为后一窗起点，没有有限差分演化。", "",
              f"本次边界设置：{config['boundary_condition']}。默认 free 允许图像边框的预测参与优化；fixed 可恢复初始边界固定行为。", "",
              ("物理损失与标注监督可能冲突，本结果为物理约束下的监督分割近似，不宣称严格 PDE 收敛或达到稳态。物理残差仅对应原始神经场，列于下表。"
               if config["gt_supervised"] else
               "GT 仅在分割完成后用于评价；演化网络只最小化 DRLSE 物理残差。有限训练给出近似解，不宣称严格 PDE 收敛或达到稳态。物理残差仅对应原始神经场，列于下表。"), "",
              "## 参数", "", json.dumps(config, indent=2, ensure_ascii=False), "",
              "## 独立残差审计", "",
              "| 窗 | 窄带残差 RMS | 冻结初始 RMS | 相对残差 | 改变符号像素 |", "|---|---:|---:|---:|---:|"]
    for row in audits["windows"]:
        b = row["narrow_band"]
        lines.append(f"| {row['window']} | {b['residual_rms']:.6g} | {b['frozen_initial_residual_rms']:.6g} | {b['relative_residual']:.4f} | {row['changed_sign_pixels']} |")
    lines += ["", "相对残差=RMS(phi_t-RHS)/(RMS(phi_t)+RMS(RHS))；冻结初始场残差为 −RHS。", "",
              "## 输出文件", "", "- segmentation_mask.png：最终前景二值掩膜，前景 255，背景 0。",
              "- raw_segmentation_mask.png：原始神经预测 phi_final<0，未经过前景整理。",
              "- segmentation_sdf.npy：最终掩膜的有符号距离场，负值对应最终前景。",
              "- segmented_image.png：保留前景的 RGB 图像。", "- contour_overlay.png：红色预测轮廓。",
              "- prediction_vs_gt.png：红色预测，绿色 GT。", "- result_summary.png：结果对照总图。",
              "- raw_vs_final.png、raw_contour_overlay.png：原始与最终掩膜对照、原始轮廓。",
              "- phi_final.npy：原始神经水平集，未被后处理覆盖。", "- model.pt：所有神经网络权重，对应原始神经场。",
              "- metrics.json、physics_audit.json、training_history.csv：指标、残差与训练记录。", "",
              f"复现（主要参数；完整参数见 config.json）：python pinn_drlse.py --origin {config['arguments']['origin']} --gt {config['arguments']['gt']} --output {out} --supervision {config['arguments']['supervision']} --init-polarity {config['arguments']['init_polarity']} --foreground-mode {post['mode']} --boundary {config['arguments'].get('boundary', 'free')} --gt-weight {config['arguments']['gt_weight']} --gt-dice-weight {config['arguments'].get('gt_dice_weight', 1.)} --sigma {config['arguments']['sigma']} --windows {config['arguments']['windows']} --dt {config['arguments']['dt']}", ""]
    (out/"README_zh.md").write_text("\n".join(lines), encoding="utf-8")


def load_ground_truth(gt_path, expected_shape):
    """Binary white foreground and signed distance, used for terminal supervision."""
    mask = np.asarray(Image.open(gt_path).convert("L")) > 127
    if mask.shape != tuple(expected_shape):
        raise ValueError(f"GT shape {mask.shape} differs from image {expected_shape}: {gt_path}")
    if not mask.any():
        sdf = np.full(mask.shape, max(mask.shape), np.float32)
    elif mask.all():
        sdf = np.full(mask.shape, -max(mask.shape), np.float32)
    else:
        sdf = (ndi.distance_transform_edt(~mask)-ndi.distance_transform_edt(mask)).astype(np.float32)
    return {"mask": mask, "sdf": sdf}

def terminal_supervised_loss(phi, labels, target_sdf, epsilon=1.5,
                             sdf_weight=0.1, distance_scale=8.0, dice_weight=1.0):
    """Terminal pixel classification, foreground overlap, and boundary distance."""
    logits = -phi/epsilon
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities*labels).sum()
    dice = 1-(2*intersection+1)/(probabilities.sum()+labels.sum()+1)
    band = (target_sdf.abs() < 16).to(phi.dtype)
    errors = F.smooth_l1_loss(phi/distance_scale, target_sdf/distance_scale, reduction="none")
    distance = (errors*band).sum()/band.sum().clamp_min(1)
    return bce+dice_weight*dice+sdf_weight*distance, {"bce": bce, "sdf": distance, "dice": dice}

def make_supervision_pool(field, supervision, rng, count, device):
    h, w = supervision["mask"].shape
    # Uniform pixel sampling includes exact border pixels. Small images use the
    # whole grid; larger ones use a uniform subset. No GT content chooses points.
    pixel_ids = np.arange(h*w) if h*w <= count else rng.choice(h*w, count, replace=False)
    ys, xs = np.divmod(pixel_ids, w)
    xy = np.column_stack((2*xs/(w-1)-1, 2*ys/(h-1)-1)).astype(np.float32)
    x = torch.from_numpy(xy).to(device)
    with torch.no_grad():
        base = torch.cat([field(x[i:i+16384]) for i in range(0, len(x), 16384)])
    # These are loss targets only, never initialization or network input features.
    labels = supervision["mask"][ys, xs].astype(np.float32)
    target = supervision["sdf"][ys, xs].astype(np.float32)
    return x, base, torch.from_numpy(labels[:, None]).to(device), torch.from_numpy(target[:, None]).to(device)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origin", help="Single-image input; omit to process --data-dir")
    p.add_argument("--gt", help="Single-image GT")
    p.add_argument("--data-dir", default=str(ROOT/"Training_data"))
    p.add_argument("--gt-dir", default=str(ROOT/"Training_GT"))
    p.add_argument("--output", help="Default: results_batch, or results_pinn for --origin")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--supervision", choices=("terminal", "none"), default="terminal",
                   help="terminal (default): GT only in final-time loss; none: evaluation only; initialization is always image-only")
    p.add_argument("--gt-weight", type=float, default=5.)
    p.add_argument("--gt-dice-weight", type=float, default=1.,
                   help="Foreground overlap term within the terminal GT loss")
    p.add_argument("--gt-sdf-weight", type=float, default=.1)
    p.add_argument("--gt-distance-scale", type=float, default=8.)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip completed images only when inputs and training configuration match")
    p.add_argument("--limit", type=int, help="Process the first N sorted image pairs")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--sigma", type=float, default=1.5,
                   help="Gaussian scale in pixels for threshold initialization and edge indicator")
    p.add_argument("--init-polarity", choices=("auto", "dark", "bright"), default="auto",
                   help="Otsu foreground: auto uses the class less present on the border; never uses GT")
    p.add_argument("--boundary", choices=("free", "fixed"), default="free",
                   help="free: allow learning at image borders; fixed: preserve initial border values")
    p.add_argument("--foreground-mode", choices=("single", "all"), default="single",
                   help="single: keep largest foreground component and fill holes; all: export raw prediction")
    p.add_argument("--mu", type=float, default=.2)
    p.add_argument("--lam", type=float, default=5.)
    p.add_argument("--alpha", type=float, default=-.5)
    p.add_argument("--epsilon", type=float, default=1.5)
    p.add_argument("--eta", type=float, default=1e-6)
    # The window duration and count set the total physical evolution time.
    p.add_argument("--dt", type=float, default=20.)
    p.add_argument("--windows", type=int, default=1)
    p.add_argument("--initial-steps", type=int, default=5000)
    p.add_argument("--window-steps", type=int, default=4000)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--pool", type=int, default=60000)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args(argv)
    for name in ("sigma", "epsilon", "eta", "dt", "windows", "initial_steps",
                 "window_steps", "width", "batch", "pool", "lr", "gt_distance_scale"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.gt_weight < 0 or args.gt_sdf_weight < 0 or args.gt_dice_weight < 0:
        p.error("GT loss weights must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if bool(args.origin) != bool(args.gt):
        p.error("--origin and --gt must be provided together")
    return args

def run_image(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uses_gt = args.supervision == "terminal" and args.gt_weight > 0
    config = dict(arguments=vars(args), device=str(device), torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  origin_sha256=hashlib.sha256(Path(args.origin).read_bytes()).hexdigest(),
                  gt_sha256=hashlib.sha256(Path(args.gt).read_bytes()).hexdigest(),
                  physical_coordinates="original image pixels", total_evolution_time=args.dt*args.windows,
                  boundary_condition=("free: spatial border values are learned; no fixed Dirichlet condition"
                                      if args.boundary == "free" else
                                      "hard Dirichlet: phi(border,t)=phi(border,0)"),
                  initial_condition="hard: phi(x,y,0)=image_threshold_initial_field",
                  architecture="Fourier-tanh MLP plus differentiable initial-phi features; time windows",
                  gt_supervised=uses_gt, gt_constraint_time=args.dt*args.windows if uses_gt else None,
                  gt_used_for_initialization=False, initialization_source="original_image_otsu",
                  gt_training_usage="terminal_loss_targets_only" if uses_gt else "none",
                  supervision_sampling="uniform_pixel_grid_label_independent" if uses_gt else None,
                  foreground_mode=args.foreground_mode, gt_used_for_postprocessing=False,
                  metric_scope="training_fit; not held-out validation" if uses_gt else "reference_evaluation_only",
                  loss="L_pde + gt_weight * (BCE(-phi(T)/epsilon,GT) + gt_dice_weight * soft_Dice + gt_sdf_weight * L_sdf_band)" if uses_gt else "L_pde")
    rgb, sdf, g, gx, gy, metadata = prepare_image(args.origin, out, args.sigma, args.init_polarity)
    config["image_preparation"] = metadata
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("CONFIG "+json.dumps(config), flush=True)
    if args.prepare_only:
        return None
    started = time.perf_counter()
    field, initial_phi, initial_audit, initial_history = pretrain_initial(sdf, out, args, device)
    save_checkpoint(field, out/"initial_model.pt", args, rgb.shape[1], rgb.shape[0])
    # Finish image-only initialization before loading any GT training targets.
    supervision = load_ground_truth(args.gt, sdf.shape) if uses_gt else None
    phi, history, windows = train_windows(field, initial_phi, g, gx, gy, out, args, device, supervision)
    audit = dict(initial_fit=initial_audit, windows=windows, total_training_seconds=time.perf_counter()-started)
    return evaluate_and_export(rgb, phi, initial_phi, args.gt, out, config, audit, initial_history+history)

def main(argv=None):
    from batch_pinn_drlse import run_batch
    args = parse_args(argv)
    if args.origin:
        args.origin, args.gt = str(Path(args.origin).resolve()), str(Path(args.gt).resolve())
        args.output = str(Path(args.output or ROOT/"results_pinn").resolve())
        # Use the same header checks as batch mode, without resizing GT.
        from batch_pinn_drlse import validate_pair
        validate_pair(Path(args.origin), Path(args.gt))
        run_image(args)
        return 0
    args.output = str(Path(args.output or ROOT/"results_batch").resolve())
    return run_batch(args, run_image)

if __name__ == "__main__":
    raise SystemExit(main())

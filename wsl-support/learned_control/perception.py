import cv2
import numpy as np
import torch
from torch import nn
from vision_mpc.perception import Detection, DetectionError, components, pixel_to_world


class PixelNet(nn.Module):
    """Small learned 1x1 CNN for the simulator's RGB semantic classes."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 12, 1), nn.ReLU(), nn.Conv2d(12, 12, 1),
                                    nn.ReLU(), nn.Conv2d(12, 6, 1))

    def forward(self, pixels):
        return self.layers(pixels*2-1)

    def detect(self, rgb):
        small = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_AREA)
        inputs = torch.from_numpy(small.copy()).permute(2, 0, 1).float()[None]/255
        with torch.no_grad():
            labels = self(inputs).argmax(1)[0].numpy()
        positions, pixels = {}, {}
        for name, label in (('hunter', 1), ('evader', 2), ('goal', 3)):
            found = components(labels == label, min_area=7, max_area=1000)
            if len(found) != 1:
                raise DetectionError(f'learned CNN {name}: expected one object, found {len(found)}')
            pixels[name] = found[0][0]*2
            positions[name] = pixel_to_world(found[0][0], 400, 400)
        found = components(labels == 4, min_area=50, max_area=1000)
        if len(found) != 3:
            raise DetectionError(f'learned CNN obstacles: expected three, found {len(found)}')
        found.sort(key=lambda pair: (pair[0][0], pair[0][1]))
        centers = np.array([pair[0] for pair in found])
        radii = np.sqrt(np.array([pair[1] for pair in found])/np.pi)*2/400
        pixels['obstacles'] = centers*2
        return Detection(positions['evader'], positions['hunter'], positions['goal'],
                         pixel_to_world(centers, 400, 400), radii, pixels)


def visual_labels(rgb):
    # Training labels come exclusively from RGB, not simulation state.
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    labels = np.zeros(rgb.shape[:2], dtype=np.int64)
    labels[(r > g+25) & (r > b+25) & (r > 215)] = 1
    labels[(g > r+25) & (g > b+25) & (g > 215)] = 2
    labels[(r > g+25) & (b > g+25) & (abs(r-b) < 15) & (r > 180)] = 3
    labels[np.maximum(np.maximum(r, g), b) < 100] = 4
    labels[(r > 150) & (g < 140) & (b < 140) & (r > b+40)] = 5
    return labels


def fit_pixel_net(frames, output, seed):
    rng = np.random.default_rng(seed)
    by_class = [[] for _ in range(6)]
    for frame in frames:
        image = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_AREA)
        labels = visual_labels(image)
        for label in range(6):
            samples = np.unique(image[labels == label], axis=0)
            if len(samples):
                by_class[label].append(samples)
    colors, classes = [], []
    for label, arrays in enumerate(by_class):
        unique = np.unique(np.concatenate(arrays), axis=0)
        pixels = unique[rng.integers(0, len(unique), 3000)]
        colors.append(pixels); classes.extend([label]*len(pixels))
    x = torch.from_numpy(np.concatenate(colors)).float()[:, :, None, None]/255
    y = torch.tensor(classes, dtype=torch.long)
    model = PixelNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.008)
    history = []
    for step in range(1000):
        indices = torch.from_numpy(rng.integers(0, len(y), 2048))
        logits = model(x[indices])[:, :, 0, 0]
        loss = nn.functional.cross_entropy(logits, y[indices])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0 or step == 999:
            history.append({'step': step+1, 'loss': loss.item(),
                            'balanced_pixel_accuracy': (logits.argmax(1) == y[indices]).float().mean().item()})
    torch.save(model.state_dict(), output/'perception.pt')
    return model.eval(), history

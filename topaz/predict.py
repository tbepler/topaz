from __future__ import absolute_import, print_function, division

import torch


def batches(X, batch_size=1):
    batch = []
    for x in X:
        batch.append(torch.from_numpy(x).float())
        if len(batch) >= batch_size:
            batch = torch.stack(batch, 0)
            yield batch
            batch = []
    if len(batch) > 0:
        batch = torch.stack(batch, 0)
        yield batch


def score_stream(model, images, use_cuda=False, batch_size=1):
    with torch.no_grad():
        for x in batches(images, batch_size=batch_size):
            x = x.unsqueeze(1)
            if use_cuda:
                x = x.cuda()
            logits = model(x).squeeze(1).cpu().numpy()
            for i in range(len(logits)):
                yield logits[i]


def score(model, images, use_cuda=False, batch_size=1):
    scores = []
    for y in score_stream(model, images, use_cuda=use_cuda, batch_size=batch_size):
        scores.append(y)
    return scores







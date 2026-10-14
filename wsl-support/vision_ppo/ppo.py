import numpy as np
import torch


def update(model, optimizer, data, cfg):
    obs, latent, old_logp, advantage, returns = [torch.as_tensor(x, dtype=torch.float32)
                                               for x in data]
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    stats = []
    stop = False
    for epoch in range(cfg['epochs']):
        for indices in np.array_split(np.random.permutation(len(obs)),
                                      max(1, int(np.ceil(len(obs)/cfg['batch_size'])))):
            dist = model.distribution(obs[indices])
            logp = model.log_prob(dist, latent[indices])
            logratio = logp-old_logp[indices]
            ratio = logratio.exp()
            kl = ((ratio-1)-logratio).mean()
            if kl.item() > cfg['target_kl']:
                stop = True
                break
            policy = -torch.minimum(ratio*advantage[indices],
                                    ratio.clamp(1-cfg['clip'], 1+cfg['clip'])*advantage[indices]).mean()
            value = .5*(model.value(obs[indices])-returns[indices]).square().mean()
            # Monte Carlo differential entropy of the bounded action policy.
            fresh = dist.rsample()
            entropy = -model.log_prob(dist, fresh).mean()
            loss = policy + cfg['value_coef']*value - cfg['entropy_coef']*entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            optimizer.step()
            stats.append([policy.item(), value.item(), entropy.item(), kl.item()])
        if stop:
            break
    result = dict(zip(('policy_loss', 'value_loss', 'entropy', 'kl'),
                      np.mean(stats, axis=0).tolist() if stats else [0.]*4))
    result['epochs'] = epoch+1
    return result

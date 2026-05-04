import torch
import torch.nn as nn

def check_data(tensor, name):
    if tensor is None:
        return
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

def _ensure_col(x):
    if x is None:
        return x
    if x.dim() == 1:
        return x.unsqueeze(1)
    return x

def _call_get_log_prob(policy_net, states, actions, next_states, index_agent, global_states, global_actions):

    f = policy_net.get_log_prob
    last_err = None

    tries = [
        lambda: f(states, actions, next_states, index_agent, global_states, global_actions),
        lambda: f(states, actions, index_agent, global_states, global_actions),
        lambda: f(states, actions, next_states, index_agent, global_states),
        lambda: f(states, actions, index_agent, global_states),
        lambda: f(states, actions, global_states),
        lambda: f(states, actions),
    ]
    for call in tries:
        try:
            return call()
        except TypeError as e:
            last_err = e

    raise last_err


def ppo_step(policy_net, value_net, optimizer_p, optimizer_v,
             states, actions, next_states, returns, old_log_probs,
             advantages, ppo_clip_ratio, value_l2_reg, index_agent,
             global_states, global_actions,
             log_ratio_clip: float = 20.0,
             max_grad_norm_v: float = 40.0,
             max_grad_norm_p: float = 40.0):

    returns = _ensure_col(returns).detach()
    old_log_probs = _ensure_col(old_log_probs).detach()
    advantages = _ensure_col(advantages).detach()

    check_data(states, "states")
    check_data(actions, "actions")
    check_data(next_states, "next_states")
    check_data(returns, "returns")
    check_data(old_log_probs, "old_log_probs")
    check_data(advantages, "advantages")
    check_data(global_states, "global_states")
    check_data(global_actions, "global_actions")

    values_pred = _ensure_col(value_net(states))
    value_loss = ((values_pred - returns) ** 2).mean()

    if value_l2_reg and value_l2_reg > 0:
        l2 = 0.0
        for p in value_net.parameters():
            l2 = l2 + p.pow(2).sum()
        value_loss = value_loss + value_l2_reg * l2

    optimizer_v.zero_grad(set_to_none=True)
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=max_grad_norm_v)
    optimizer_v.step()

    new_log_probs = _call_get_log_prob(
        policy_net=policy_net,
        states=states,
        actions=actions,
        next_states=next_states,
        index_agent=index_agent,
        global_states=global_states,
        global_actions=global_actions,
    )
    new_log_probs = _ensure_col(new_log_probs)

    log_ratio = (new_log_probs - old_log_probs).clamp(-log_ratio_clip, log_ratio_clip)
    ratio = torch.exp(log_ratio)

    sur1 = ratio * advantages
    sur2 = torch.clamp(ratio, 1 - ppo_clip_ratio, 1 + ppo_clip_ratio) * advantages
    policy_loss = -torch.min(sur1, sur2).mean()

    optimizer_p.zero_grad(set_to_none=True)
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=max_grad_norm_p)
    optimizer_p.step()

    return value_loss.detach(), policy_loss.detach()

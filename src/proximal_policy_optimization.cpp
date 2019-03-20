#include "proximal_policy_optimization.h"

auto PPO::returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT
{
    // Compute the returns.
    torch::Tensor gae = torch::zeros({1}, torch::kFloat32);
    VT returns(rewards.size(), torch::zeros({1}, torch::kFloat32));

    for (uint i=rewards.size();i-- >0;) // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
    {
        // Advantage.
        auto delta = rewards[i] + gamma*vals[i+1]*(1-dones[i]) - vals[i];
        gae = delta + gamma*lambda*(1-dones[i])*gae;

        returns[i] = gae + vals[i];
    }

    return returns;
}

auto PPO::update(ActorCritic& ac,
                 torch::Tensor& left_states,
                 torch::Tensor& right_states,
                 torch::Tensor& actions,
                 torch::Tensor& log_probs,
                 torch::Tensor& returns,
                 torch::Tensor& advantages, 
                 OPT& opt, 
                 uint steps, uint epochs, uint mini_batch_size, double beta, double clip_param) -> void
{
    for (uint e=0;e<epochs;e++)
    {
        // Generate random indices.
        std::vector<uint> idx;
        idx.reserve(mini_batch_size);

        for (uint b=0;b<mini_batch_size;b++) {

            idx.push_back(std::uniform_int_distribution<uint>(0, steps-1)(re));
        }

        for (auto& i: idx)
        {
            auto av = ac->forward(left_states[i], right_states[i]); // action value pairs
            auto action = std::get<0>(av);
            auto entropy = ac->entropy();
            auto new_log_prob = ac->log_prob(actions[i]);

            auto old_log_prob = log_probs[i];
            auto ratio = (new_log_prob - old_log_prob).exp();
            auto surr1 = ratio*advantages[i];
            auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param)*advantages[i];

            auto val = std::get<1>(av);
            auto actor_loss = -torch::min(surr1, surr2);
            auto critic_loss = (returns[i]-val).pow(2);

            auto loss = 0.5*critic_loss+actor_loss-beta*entropy;

            opt.zero_grad();
            loss.backward();
            opt.step();
        }
    }
}

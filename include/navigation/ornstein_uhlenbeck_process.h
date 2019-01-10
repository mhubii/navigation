#ifndef ORNSTEIN_UHLENBECK_PROCESS_H_
#define ORNSTEIN_UHLENBECK_PROCESS_H_

#include <torch/torch.h>

// Ornstein-Uhlenbeck process.
class OUProcess {

    public:

        OUProcess(at::IntList size, double mu, double theta, double sigma);

        void Reset();

        torch::Tensor Sample();

    private:

        // Parameters.
        at::IntList size_;
        torch::Tensor mu_;
        torch::Tensor state_;
        double theta_;
        double sigma_;
};

#endif

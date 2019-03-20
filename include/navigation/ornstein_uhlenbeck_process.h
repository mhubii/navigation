#ifndef ORNSTEIN_UHLENBECK_PROCESS_H_
#define ORNSTEIN_UHLENBECK_PROCESS_H_

#include <torch/torch.h>

// Ornstein-Uhlenbeck process.
class OUProcess {

    public:

        OUProcess(torch::IntArrayRef size, float mu, float theta, float sigma);

        void Reset();

        torch::Tensor Sample();

        void print();

    private:

        // Parameters.
        torch::Tensor mu_;
        torch::Tensor state_;
        float theta_;
        float sigma_;
};

#endif

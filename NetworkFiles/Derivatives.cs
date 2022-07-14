using System;
using static NeatNetwork.Libraries.Cost;
using static NeatNetwork.Libraries.Activation;

namespace NeatNetwork.Libraries
{
    public static class Derivatives
    {

        /// <summary>
        /// This function is used for supervised learning only
        /// </summary>
        public static double[] DerivativeOf(double[] output, double[] label, CostFunctions costFunction)
        {
            double[] costGrads = new double[label.Length];
            for (int i = 0; i < label.Length; i++)
                costGrads[i] = DerivativeOf(output[i], label[i], costFunction);

            return costGrads;
        }

        /// <param name="expected">In case of Reinforcement learning/logLikelyhood expected is reward</param>
        public static double DerivativeOf(double neuronActivation, double expected, CostFunctions costFunction)
        {
            if (double.IsNaN(expected))
            {
                return 0;
            }
            switch (costFunction)
            {
                case CostFunctions.BinaryCrossEntropy:
                    throw new NotImplementedException();
                case CostFunctions.SquaredMean:
                    return SquaredMeanErrorDerivative(neuronActivation, expected);
                case CostFunctions.logLikelyhoodTerm:
                    return LogLikelyhoodTermDerivative(neuronActivation, expected);
                default:
                    throw new NotImplementedException();
            }
        }

        public static double LogLikelyhoodTermDerivative(double output, double reward) => -(1 / output * Math.Log(10)) * -Math.Log10(output) + reward;

        public static double SquaredMeanErrorDerivative(double neuronOutput, double expectedOutput) => 2 * (neuronOutput - expectedOutput);

        //public static double BinaryCrossEntropyDerivative(double neuronOutput, double expectedOutput) =>  /


        public static double DerivativeOf(double neuronLinear, ActivationFunctions activation)
        {
            double output;
            switch (activation)
            {
                case ActivationFunctions.Relu:
                    output = ReluDerivative(neuronLinear);
                    break;
                case ActivationFunctions.Sigmoid:
                    output = SigmoidDerivative(neuronLinear);
                    break;
                case ActivationFunctions.Tanh:
                    output = TanhDerivative(neuronLinear);
                    break;
                case ActivationFunctions.Sine:
                    output = SinDerivative(neuronLinear);
                    break;
                default:
                    throw new NotImplementedException();
            }
            return output;
        }

        public static double SigmoidDerivative(double neuronActivation) => SigmoidActivation(neuronActivation) * (1 - SigmoidActivation(neuronActivation));

        /// <param name="connectedNeuronActivation">Activation Connected to the weigth that is being computed</param>
        public static double LinearFunctionDerivative(double connectedNeuronActivation) => connectedNeuronActivation;

        public static int ReluDerivative(double neuronActivation) => 1 * Convert.ToInt32(neuronActivation >= 0);

        public static double TanhDerivative(double neuronActivation) => 1 - Math.Pow(Activation.TanhActivation(neuronActivation), 2);

        public static double SinDerivative(double neuronActivation) => Math.Cos(neuronActivation);

        public static double LnDerivative(double neuronActivation) => 1 / neuronActivation;

        public static double MultiplicationDerivative(double a, double aDerivative, double b, double bDerivative) => a * aDerivative + b * bDerivative;

        public static double SumDerivative(double aDerivative, double bDerivative) => aDerivative + bDerivative;

    }
}

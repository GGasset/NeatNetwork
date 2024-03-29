﻿using System;
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

        /// <summary>
        /// This function is made for reinforcement learning
        /// </summary>
        /// <param name="output"></param>
        /// <param name="reward"></param>
        /// <returns></returns>
        public static double[] DerivativeOf(double[] output, double reward)
        {
            double[] costGrads = new double[output.Length];
            for (int i = 0; i < costGrads.Length; i++)
                costGrads[i] = DerivativeOf(output[i], reward, CostFunctions.logLikelyhoodTerm);
            return costGrads;
        }

        /// <param name="expected">In case of Reinforcement learning/logLikelyhood expected is reward</param>
        public static double DerivativeOf(double neuronActivation, double expected, CostFunctions costFunction)
        {
            switch (costFunction)
            {
                case CostFunctions.BinaryCrossEntropy:
                    return BinaryCrossEntropyDerivative(neuronActivation, expected);
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

        public static double BinaryCrossEntropyDerivative(double neuronOutput, double expectedOutput)
        {
            throw new NotImplementedException(); 

            double DLeft = neuronOutput + (Math.Log10(expectedOutput) + (1 / (expectedOutput * Math.Log(10))));
            double DRight = (1 - neuronOutput) * -neuronOutput + Math.Log10(1 - expectedOutput) * (1 / ((1 - expectedOutput) * Math.Log(10)));
            double D = DLeft + DRight;
            return D;
        }


        public static double DerivativeOf(double neuronLinear, ActivationFunctions activation)
        {
            double output;
            switch (activation)
            {
                case ActivationFunctions.Relu:
                    output = Relu(neuronLinear);
                    break;
                case ActivationFunctions.Sigmoid:
                    output = Sigmoid(neuronLinear);
                    break;
                case ActivationFunctions.Tanh:
                    output = Tanh(neuronLinear);
                    break;
                case ActivationFunctions.Sine:
                    output = Sin(neuronLinear);
                    break;
                default:
                    throw new NotImplementedException();
            }
            return output;
        }

        public static double Sigmoid(double neuronActivation) => Activation.Sigmoid(neuronActivation) * (1 - Activation.Sigmoid(neuronActivation));

        /// <param name="connectedNeuronActivation">ActivationFunction Connected to the weigth that is being computed</param>
        public static double LinearFunctionWeight(double connectedNeuronActivation) => connectedNeuronActivation;

        public static double LinearFunctionPreviousActivation(double connectedWeight) => connectedWeight;

        public static int Relu(double activationFunctionInput) => 1 * Convert.ToInt32(activationFunctionInput >= 0);

        public static double Tanh(double activationFunctionInput) => 1 - Math.Pow(Activation.Tanh(activationFunctionInput), 2);

        public static double Sin(double activationFunctionInput) => Math.Cos(activationFunctionInput);

        public static double Ln(double activationFunctionInput) => 1 / activationFunctionInput;

        public static double Multiplication(double a, double aDerivative, double b, double bDerivative) => a * aDerivative + b * bDerivative;

        public static double Sum(double aDerivative, double bDerivative) => aDerivative + bDerivative;

    }
}

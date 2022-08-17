using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.Libraries
{
    public static class Activation
    {
        public enum ActivationFunctions
        {
            Relu,
            Sigmoid,
            Tanh,
            Sine,
            GELU,
            Ln,
        }

        public static double Activate(double input, ActivationFunctions activation)
        {
            switch (activation)
            {
                case ActivationFunctions.Relu:
                    return Relu(input);
                case ActivationFunctions.Sigmoid:
                    return Sigmoid(input);
                case ActivationFunctions.Tanh:
                    return Tanh(input);
                case ActivationFunctions.Sine:
                    return Math.Sin(input);
                case ActivationFunctions.GELU:
                    return GELUActivation(input);
                case ActivationFunctions.Ln:
                    return Math.Log(input);
                default:
                    throw new NotImplementedException();
            }
        }

        public static double Relu(double input) => Math.Max(0, input);

        public static double Sigmoid(double input) => 1.0 / (1 + Math.Pow(Math.E, -input));

        public static double GELUActivation(double input) => .5 * input * (1 + Tanh(Math.Sqrt(2 / Math.PI) * (input + 0.044715 * Math.Pow(input, 3))));

        public static double Tanh(double input) => (Math.Exp(input) - Math.Exp(-input)) / (Math.Exp(input) + Math.Exp(-input));
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.Libraries
{
    public class Cost
    {
        public enum CostFunctions
        {
            SquaredMean,
            BinaryCrossEntropy,
            logLikelyhoodTerm,
        }

        public static double GetCost(double[] networkOutput, double[] expected, CostFunctions costFunction)
        {
            double output = 0;
            for (int i = 0; i < networkOutput.Length; i++)
            {
                output += GetCost(networkOutput[i], expected[i], costFunction);
            }
            return output / networkOutput.Length;
        }

        public static double GetCost(double neuronOutput, double expected, CostFunctions costFunction)
        {
            switch (costFunction)
            {
                case CostFunctions.SquaredMean:
                    return Math.Pow(neuronOutput - expected, 2);
                case CostFunctions.BinaryCrossEntropy:
                    throw new NotImplementedException();
                    break;
                case CostFunctions.logLikelyhoodTerm:
                    return LogLikelyhoodLoss(expected, neuronOutput);
                default:
                    throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Get reinforcement learning cost
        /// </summary>
        /// <param name="reward"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public static double[] GetCost(double reward, double[] output)
        {
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = LogLikelyhoodLoss(reward, output[i]);
            }

            return output;
        }

        public static double LogLikelyhoodLoss(double reward, double output) => 1 - reward * Math.Log(1 - output) + reward * Math.Log(reward);
    }
}

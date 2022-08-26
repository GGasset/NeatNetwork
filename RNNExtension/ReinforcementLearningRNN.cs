using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;
using NeatNetwork.Libraries;

namespace NeatNetwork
{
    internal class ReinforcementLearningRNN
    {
        public RNN n;
        public double LearningRate;

        List<List<NeuronExecutionValues[]>> neuronExecutionValues;
        List<List<double[]>> neuronOutputs;

        List<double> RewardHistory;
        double CurrentDefaultReward;

        public double[] Execute(double[] input)
        {
            var output = n.Execute(input, out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronExecutionOutputs);
            neuronExecutionValues.Add(networkExecutionValues);
            neuronOutputs.Add(neuronExecutionOutputs);
            RewardHistory.Add(CurrentDefaultReward);

            return output;
        }

        public void TerminateAgent(bool deleteMemory = true)
        {
            int tSCount = neuronOutputs.Count;
            List<double[]> costGradients = new List<double[]>();
            for (int t = 0; t < tSCount; t++)
            {
                List<double[]> currentNeuronOutputs = neuronOutputs[t];
                double[] currentNetworkOutputs = currentNeuronOutputs[currentNeuronOutputs.Count - 1];
                costGradients.Add(Derivatives.DerivativeOf(currentNetworkOutputs, RewardHistory[t]));
            }

            var gradients = n.GetGradients(costGradients, neuronExecutionValues, neuronOutputs);
            n.SubtractGrads(gradients, LearningRate);

            if (deleteMemory)
            {
                neuronExecutionValues = new List<List<NeuronExecutionValues[]>>();
                neuronOutputs = new List<List<double[]>>();
                RewardHistory = new List<double>();
                n.DeleteMemory();
            }
        }
    }
}

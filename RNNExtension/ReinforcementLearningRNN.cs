using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;

namespace NeatNetwork
{
    internal class ReinforcementLearningRNN
    {
        public RNN n;
        List<double[]> Inputs;
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
    }
}

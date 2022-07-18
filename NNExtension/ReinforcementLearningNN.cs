using NeatNetwork.Libraries;
using NeatNetwork.NetworkFiles;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork
{
    public class ReinforcementLearningNN
    {
        internal NN n;
        private List<List<double[]>> LinearFunctions;
        private List<List<double[]>> NeuronActivations;
        private List<double[]> Outputs;
        private List<double> Rewards;
        internal double Reward;
        internal double LearningRate;

        public ReinforcementLearningNN(NN network, double learningRate)
        {
            this.LearningRate = learningRate;
            this.n = network;

            LinearFunctions = new List<List<double[]>>();
            NeuronActivations = new List<List<double[]>>();
            Outputs = new List<double[]>();
            Rewards = new List<double>();
            Reward = 0;
        }

        internal double[] Execute(double[] input)
        {
            double[] output = n.Execute(input, out List<double[]> linearFunctions, out List<double[]> neuronActivations);
            LinearFunctions.Add(linearFunctions);
            NeuronActivations.Add(neuronActivations);
            Outputs.Add(output);
            Rewards.Add(Reward);
            return output;
        }

        internal void TerminateAgent()
        {
            double[] currentCosts;
            List<GradientValues[]> currentGradient;
            double cost = 0;
            for (int i = 0; i < Rewards.Count; i++)
            {
                cost += Cost.GetCost(Outputs[i], Rewards[i]);
                currentCosts = Derivatives.DerivativeOf(Outputs[i], Rewards[i]);
                currentGradient = n.GetGradients(LinearFunctions[i], NeuronActivations[i], currentCosts);
                n.SubtractGrads(currentGradient, LearningRate);
            }
            cost /= Rewards.Count();

            LinearFunctions = new List<List<double[]>>();
            NeuronActivations = new List<List<double[]>>();
            Outputs = new List<double[]>();
            Rewards = new List<double>();
            Reward = 0;
        }

        internal void GiveReward(double reward)
        {
            Reward += reward;
        }

        internal void AddToLastReward(double deltaReward, bool addToDefaultReward)
        {
            Rewards[Rewards.Count - 1] += deltaReward;
            Reward += deltaReward * Convert.ToInt32(addToDefaultReward);
        }
    }
}

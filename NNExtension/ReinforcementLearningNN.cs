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
        public NN n;
        private List<List<double[]>> LinearFunctions;
        private List<List<double[]>> NeuronActivations;
        private List<double[]> Outputs;
        private List<double> Rewards;
        public double Reward;
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

        public double[] Execute(double[] input)
        {
            double[] output = n.Execute(input, out List<double[]> linearFunctions, out List<double[]> neuronActivations);
            LinearFunctions.Add(linearFunctions);
            NeuronActivations.Add(neuronActivations);
            Outputs.Add(output);
            Rewards.Add(Reward);
            return output;
        }

        public void TerminateAgent()
        {
            Task<double>[] tasks = new Task<double>[Rewards.Count];
            GradientWorker[] workers = new GradientWorker[Rewards.Count];
            for (int i = 0; i < Rewards.Count; i++)
            {
                workers[i] = new GradientWorker(n, Outputs, Rewards, LinearFunctions, NeuronActivations, LearningRate, i);
                tasks[i] = Task.Run(() => workers[i].WorkWithGradients());
            }

            double cost = 0;
            for (int i = 0; i < Rewards.Count; i++)
            {
                tasks[i].Wait();
                cost += tasks[i].Result;
            }
            cost /= Rewards.Count();

            LinearFunctions = new List<List<double[]>>();
            NeuronActivations = new List<List<double[]>>();
            Outputs = new List<double[]>();
            Rewards = new List<double>();
            Reward = 0;
        }

        class GradientWorker
        {
            NN n; List<double[]> Outputs; List<double> Rewards; List<List<double[]>> LinearFunctions; List<List<double[]>> NeuronActivations; double LearningRate; int i;

            public GradientWorker(NN n, List<double[]> Outputs, List<double> Rewards, List<List<double[]>> LinearFunctions, List<List<double[]>> NeuronActivations, double LearningRate, int i)
            {
                this.n = n;
                this.Outputs = Outputs;
                this.Rewards = Rewards;
                this.LinearFunctions = LinearFunctions;
                this.NeuronActivations = NeuronActivations;
                this.LearningRate = LearningRate;
                this.i = i;
            }

            public double WorkWithGradients()
            {
                double[] currentCosts;
                List<GradientValues[]> currentGradient;
                currentCosts = Derivatives.DerivativeOf(Outputs[i], Rewards[i]);
                currentGradient = n.GetGradients(LinearFunctions[i], NeuronActivations[i], currentCosts);
                n.SubtractGrads(currentGradient, LearningRate);
                double cost = Cost.GetCost(Outputs[i], Rewards[i]);
                return cost;
            }
        }

        /// <summary>
        /// Changes default reward for future executions
        /// </summary>
        /// <param name="reward"></param>
        public void DeltaReward(double reward)
        {
            Reward += reward;
        }

        /// <summary>
        /// Changes last reward
        /// </summary>
        /// <param name="deltaReward"></param>
        /// <param name="addToDefaultReward"></param>
        public void GiveReward(double deltaReward, bool addToDefaultReward = true)
        {
            Rewards[Rewards.Count - 1] += deltaReward;
            Reward += deltaReward * Convert.ToInt32(addToDefaultReward);
        }

        public void SetLastReward(double reward, bool setDefaultReward = true)
        {
            Rewards[Rewards.Count - 1] = reward;
            Reward = reward;
        }
    }
}

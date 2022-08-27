﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;
using NeatNetwork.Libraries;

namespace NeatNetwork
{
    public class ReinforcementLearningRNN
    {
        public RNN n;
        public double LearningRate;

        List<List<NeuronExecutionValues[]>> neuronExecutionValues;
        List<List<double[]>> neuronOutputs;

        List<double> RewardHistory;

        public ReinforcementLearningRNN(RNN n, double learningRate = .5)
        {
            this.n = n;
            LearningRate = learningRate;

            neuronExecutionValues = new List<List<NeuronExecutionValues[]>>();
            neuronOutputs = new List<List<double[]>>();
            RewardHistory = new List<double>();
        }

        /// <summary>
        /// For proper training you must give a reward after calling this function
        /// </summary>
        public double[] Execute(double[] input)
        {
            var output = n.Execute(input, out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronExecutionOutputs);
            neuronExecutionValues.Add(networkExecutionValues);
            neuronOutputs.Add(neuronExecutionOutputs);

            return output;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="deleteMemory">If set to false training data and memories won't be deleted, this can cause problems at training because maybe the agent wouldn't be doung the same after being trained, so don't abuse it1</param>
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


        public void GiveReward(double reward)
        {
            RewardHistory.Add(reward);
        }
    }
}

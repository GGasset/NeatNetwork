using System;
using NeatNetwork.NetworkFiles;
using NeatNetwork.Libraries;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork
{
    public class NN
    {
        internal Activation.ActivationFunctions Activation;
        /// <summary>
        /// Input layer isn't instatiated
        /// </summary>
        internal List<List<Neuron>> Neurons;
        internal List<List<double>> MaxMutationGrid;
        internal double MaxMutationOfMutationValues;
        internal double MaxMutationOfMutationValueOfMutationValues;
        internal double MutationProbability;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layerLengths">Layer 0 in input layer and last layer is output layer</param>
        /// <param name="weightClosestTo0">If both max/min weight are positive or negative it will become useless</param>
        public NN(int[] layerLengths, Activation.ActivationFunctions activation, double maxWeight = 1.5, double minWeight = -1.5, double weightClosestTo0 = 0.37, double startingBias = 1, double mutationChance = .10, double initialValueForMaxMutation = .27, double maxMutationOfMutationValues = .2, double maxMutationOfMutationValueOfMutationValues = .05)
        {
            Neurons = new List<List<Neuron>>();
            MaxMutationGrid = new List<List<double>>();

            for (int i = 1; i < layerLengths.Length; i++)
            {
                Neurons.Add(new List<Neuron>());
                MaxMutationGrid.Add(new List<double>());

                for (int j = 0; j < Math.Abs(layerLengths[i]); j++)
                {
                    Neurons[i].Add(new Neuron(i, startingBias, layerLengths[i - 1], maxWeight, minWeight, weightClosestTo0));
                    MaxMutationGrid[i].Add(initialValueForMaxMutation);
                }
            }

            this.Activation = activation;
            this.MaxMutationOfMutationValues = maxMutationOfMutationValues;
            this.MaxMutationOfMutationValueOfMutationValues = maxMutationOfMutationValueOfMutationValues;
            MutationProbability = mutationChance;
        }

        internal double[] Execute(double[] input) => Execute(input, out _, out _);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="neuronOutputs">first array corresponds to input</param>
        /// <param name="linearFunctions">first array corresponds to the next layer of input layer</param>
        /// <returns></returns>
        internal double[] Execute(double[] input, out List<double[]> neuronOutputs, out List<double[]> linearFunctions)
        {
            neuronOutputs = new List<double[]>
            {
                input
            };

            linearFunctions = new List<double[]>();
            for (int i = 0; i < Neurons.Count; i++)
            {
                int layerLength = Neurons[i].Count;
                double[] layerOutput = new double[layerLength];
                double[] layerLinears = new double[layerLength];
                for (int j = 0; j < layerLength; j++)
                {
                    layerOutput[j] = Neurons[i][j].Execute(neuronOutputs, Activation, out double linear);
                    layerLinears[j] = linear;
                }
                linearFunctions.Add(layerLinears);
                neuronOutputs.Add(layerOutput);
            }

            return neuronOutputs[neuronOutputs.Count - 1];
        }
    }
}

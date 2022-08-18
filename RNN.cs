using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;
using static NeatNetwork.Libraries.Activation;

namespace NeatNetwork
{
    public class RNN
    {
        public int Length => Neurons.Count;

        internal ActivationFunctions ActivationFunction;
        internal List<List<NeuronHolder>> Neurons;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="networkExecutionValues"></param>
        /// <returns></returns>
        public double[] Execute(double[] input, out List<List<NeuronValues>> networkExecutionValues)
        {
            networkExecutionValues = new List<List<NeuronValues>>();
            List<double[]> neuronActivations = new List<double[]>()
            {
                input
            };

            for (int i = 0; i < Length; i++)
            {
                networkExecutionValues.Add(new List<NeuronValues>());

                int layerLength = Neurons[i].Count;
                neuronActivations.Add(new double[layerLength]);

                for (int j = 0; j < layerLength; j++)
                {
                    neuronActivations[i + 1][j] = Neurons[i][j].Execute(neuronActivations, ActivationFunction, out NeuronValues neuronExecutionValues);
                    networkExecutionValues[i].Add(neuronExecutionValues);
                }
            }

            return neuronActivations[neuronActivations.Count - 1];
        }
    }
}

using NeatNetwork.Libraries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    internal class NeuronHolder
    {
        public readonly NeuronType neuronType;
        internal Neuron Neuron;
        internal LSTMNeuron LSTMNeuron;

        internal double Execute(List<double[]> previousNeuronsActivations, Activation.ActivationFunctions activationFunction, out NeuronValues outputVals)
        {
            double activation;
            switch (neuronType)
            {
                case NeuronType.Neuron:
                    activation = Neuron.Execute(previousNeuronsActivations, activationFunction, out double linearFunction);
                    outputVals = new NeuronValues(neuronType)
                    {
                        LinearFunction = linearFunction,
                        Activation = activation,
                    };
                    break;
                case NeuronType.LSTM:
                    activation = LSTMNeuron.Execute(previousNeuronsActivations, out outputVals);
                    break;
                case NeuronType.Recurrent:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException();
            }
            return activation;
        }

        internal void SubtractGrads(NeuronHolder gradients, double learningRate)
        {
            
        }

        public enum NeuronType
        {
            Neuron,
            LSTM,
            Recurrent
        }
    }
}

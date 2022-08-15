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
                    break;
                case NeuronType.LSTM:

                    break;
                case NeuronType.Recurrent:
                    break;
                default:
                    throw new NotImplementedException();
            }
            return activation;
        }

        public enum NeuronType
        {
            Neuron,
            LSTM,
            Recurrent
        }
    }
}

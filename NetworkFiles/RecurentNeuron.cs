using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    internal class RecurentNeuron
    {
        internal readonly NeuronType neuronType;
        internal Neuron Neuron;

        public enum NeuronType
        {
            Neuron,
            LSTM,
            Recurrent
        }
    }
}

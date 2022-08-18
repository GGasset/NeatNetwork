using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;
using static NeatNetwork.Libraries.Activation;

namespace NeatNetwork
{
    internal class RNN
    {
        public int Length => Neurons.Count;

        internal ActivationFunctions ActivationFunction;
        internal List<List<NeuronHolder>> Neurons;

        
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;

namespace NeatNetwork.Libraries
{
    internal class GradientsNetworkAssociation
    {
        internal List<List<NeuronHolder>> Gradients;
        internal int NetworkIndex;

        internal GradientsNetworkAssociation(List<List<NeuronHolder>> gradients, int networkIndex)
        {
            Gradients = gradients;
            NetworkIndex = networkIndex;
        }
    }
}

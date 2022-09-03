using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Groupings;
using NeatNetwork.NetworkFiles;

namespace NeatNetwork.Libraries
{
    internal class NetworkGroupGradients
    {
        internal List<int> ExecutionOrder;
        internal List<List<Connection>> outputConnectionsGradients;
        internal List<List<List<Connection>>> ConnectionsGradients;
        internal List<List<List<NeuronHolder>>> NetworksGradients;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Libraries;

namespace NeatNetwork.Groupings
{
    internal class AgroupatedNetwork
    {
        private List<double> input;
        internal List<Connection> Connections;
        public RNN n;

        internal AgroupatedNetwork(RNN n)
        {
            this.n = n;
            Connections = new List<Connection>();
            ClearInput();
        }

        internal void Connect(int connectedNetworkI, int connectedNetworkOutputLength, Range inputRange, Range outputRange, double maxWeight, double minWeight, double weightClosestTo0)
        {
            var connection = new Connection(inputRange, connectedNetworkI, outputRange, n.InputLength, connectedNetworkOutputLength, maxWeight, minWeight, weightClosestTo0);
            for (int i = 0; i < Connections.Count; i++)
            {
                if (Connections[i].ConnectedNetworkI >= connectedNetworkI)
                {
                    Connections.Insert(i, connection);
                    return;
                }
            }
            Connections.Add(connection);
        }

        internal void ClearInput() => input = new List<double>(new double[n.InputLength]);
    }
}

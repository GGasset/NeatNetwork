﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Libraries;
using NeatNetwork.NetworkFiles;

namespace NeatNetwork.Groupings
{
    public class AgroupatedNetwork
    {
        internal List<double> input;
        internal List<Connection> Connections;
        public RNN n;

        internal AgroupatedNetwork(RNN n)
        {
            this.n = n;
            Connections = new List<Connection>();
            ClearInput();
        }

        internal void PassInput(double[] wholeInput, int connectionI, int connectedNetworkOutputLength)
        {
            var connection = Connections[connectionI];
            Range inputRange = FormatRange(connectionI, connectedNetworkOutputLength, true), outputRange = FormatRange(connectionI, connectedNetworkOutputLength, false);
            for (int connectedNetworkInputI = inputRange.FromI; connectedNetworkInputI <= inputRange.ToI; connectedNetworkInputI++)
            {
                for (int networkOutputI = outputRange.FromI; networkOutputI <= outputRange.ToI; networkOutputI++)
                {
                    input[connectedNetworkInputI] += wholeInput[networkOutputI] * connection.Weights[connectedNetworkInputI - inputRange.FromI][networkOutputI - outputRange.FromI];
                }
            }
        }

        /// <summary>
        /// Connecting to -1 will mean input connected
        /// </summary>
        /// <param name="connectedNetworkI"></param>
        /// <param name="connectedNetworkOutputLength"></param>
        /// <param name="inputRange"></param>
        /// <param name="outputRange"></param>
        /// <param name="maxWeight"></param>
        /// <param name="minWeight"></param>
        /// <param name="weightClosestTo0"></param>
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

        // TODO: implement optimized search
        public bool IsConnectedTo(int networkI)
        {
            for (int i = 0; i < Connections.Count; i++)
                if (Connections[i].ConnectedNetworkI == networkI)
                    return true;

            return false;
        }

        internal Connection GetConnectionConnectedTo(int networkI) => GetConnectionConnectedTo(networkI, out _);

        // TODO implement optimized search
        internal Connection GetConnectionConnectedTo(int networkI, out int connectionI)
        {
            for (int i = 0; i < Connections.Count; i++)
                if (Connections[i].ConnectedNetworkI == networkI)
                {
                    connectionI = i;
                    return Connections[i];
                }

            throw new ArgumentException("Connection doesn't exists");
        }

        internal Range GetOutputRange(int connectionIndex, int connectedNetworkOutputLength) => FormatRange(connectionIndex, connectedNetworkOutputLength, false);

        internal Range GetInputRange(int connectionIndex, int connectedNetworkOutputLength) => FormatRange(connectionIndex, connectedNetworkOutputLength, true);

        internal Range FormatRange(int connectionI, int connectedNetworkOutputLength, bool isInputRange)
            => Connections[connectionI].FormatRange(n.InputLength, connectedNetworkOutputLength, isInputRange);
    }
}

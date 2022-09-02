using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;
using NeatNetwork.Libraries;

namespace NeatNetwork.Groupings
{
    public class NetworkGroup
    {
        internal List<AgroupatedNetwork> Networks;
        public List<int> ExecutionOrder;
        internal List<Connection> OutputConnections;

        public readonly int InputLength;
        public readonly int OutputLength;

        private double[] Output;

        public NetworkGroup(int inputLength, int outputLength)
        {
            Networks = new List<AgroupatedNetwork>();
            ExecutionOrder = new List<int>();
            OutputConnections = new List<Connection>();
            InputLength = inputLength;
            OutputLength = outputLength;
        }

        public double[] Execute(double[] input)
        {
            ClearOutput();

            List<int> inputConnectedNetworks = GetNetworksConnectedTo(-1);
            foreach (var networkI in inputConnectedNetworks)
            {
                Connection inputConnectedConnection = Networks[networkI].GetConnectionConnectedTo(-1);
                Networks[networkI].PassInput(input, inputConnectedConnection);
            }

            for (int i = 0; i < ExecutionOrder.Count; i++)
            {
                int currentExecutionNetwork = ExecutionOrder[i];

                double[] nOutput = Networks[currentExecutionNetwork].n.Execute(input, out List<NeuronExecutionValues[]> 
                    neuronExecutionValues, out List<double[]> neuronActivations);

                List<int> networksConnectedToCurrentNetwork = GetNetworksConnectedTo(currentExecutionNetwork);
                foreach (var networkIConnectedToCurrentNetwork in networksConnectedToCurrentNetwork)
                {
                    Networks[networkIConnectedToCurrentNetwork].PassInput(nOutput, Networks[networkIConnectedToCurrentNetwork].GetConnectionConnectedTo(currentExecutionNetwork));
                }

                Connection outputConnection = GetOutputConnection(currentExecutionNetwork);
                if (outputConnection != null)
                    PassOutput(nOutput, outputConnection);
            }

            return Output;
        }

        #region Gradient Learning

        /// <summary>
        /// 
        /// </summary>
        /// <param name="costGradients">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="executionValues">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="neuronActivations">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="connectionsGradients">List representing networks networkI, containing lists representing connections, containing list representing input connections, containing a list representing weights cost</param>
        /// <param name="outputConnectionsGradients">List representing connections, then group output neuron, then weight</param>
        /// <returns></returns>
        internal List<List<List<NeuronHolder>>> GetGradients(out List<List<List<List<double>>>> connectionsGradients, out List<List<List<double>>> outputConnectionsGradients, List<List<double[]>> costGradients, List<List<List<NeuronExecutionValues[]>>> executionValues, List<List<List<double[]>>> neuronActivations)
        {
            int tSCount = costGradients.Count;

            // NetworkOutputGradients is a list representing execution order containing other list representing time containing a cost arrays for networks
            List<List<double[]>> networksOutputCostGradients = new List<List<double[]>>();
            for (int i = 0; i < ExecutionOrder.Count; i++)
            {
                networksOutputCostGradients.Add(new List<double[]>());
                for (int t = 0; t < tSCount; t++)
                {
                    networksOutputCostGradients[i].Add(new double[Networks[ExecutionOrder[i]].n.OutputLength]);
                }
            }

            connectionsGradients = new List<List<List<List<double>>>>();
            for (int networkI = 0; networkI < Networks.Count; networkI++)
            {
                connectionsGradients.Add(new List<List<List<double>>>());
                for (int connectionI = 0; connectionI < Networks[networkI].Connections.Count; connectionI++)
                {
                    connectionsGradients[networkI].Add(new List<List<double>>());

                    Connection cConnection = Networks[networkI].Connections[connectionI];
                    for (int neuronI = 0; neuronI < cConnection.InputRange.Length; neuronI++)
                    {
                        connectionsGradients[networkI][connectionI].Add(new List<double>());
                        for (int weightI = 0; weightI < cConnection.ConnectedOutputRange.Length; weightI++)
                        {
                            connectionsGradients[networkI][connectionI][neuronI].Add(0);
                        }
                    }
                }
            }

            outputConnectionsGradients = new List<List<List<double>>>();
            for (int connectionI = 0; connectionI < OutputConnections.Count; connectionI++)
            {
                outputConnectionsGradients.Add(new List<List<double>>());
                for (int neuronI = 0; neuronI < OutputConnections[connectionI].InputRange.Length; neuronI++)
                {
                    outputConnectionsGradients[connectionI].Add(new List<double>());
                    for (int weightI = 0; weightI < OutputConnections[connectionI].ConnectedOutputRange.Length; weightI++)
                    {
                        outputConnectionsGradients[connectionI][neuronI].Add(0);
                    }
                }
            }

            // TODO: Pass output costs to output connected networks

            for (int i = ExecutionOrder.Count - 1; i >= 0; i--)
            {
                int currentExecutionNetworkI = ExecutionOrder[i];

                // Putting relevant values in a better formatted manner for training, a list representing time
                List<double[]> currentCostGradients = new List<double[]>();
                List < List<NeuronExecutionValues[]>> currentExecutionValues = new List<List<NeuronExecutionValues[]>>();
                List < List<double[]>> currentNeuronActivations = new List<List<double[]>>();

                for (int t = 0; t < tSCount; t++)
                {
                    currentCostGradients.Add(costGradients[t][currentExecutionNetworkI]);
                    currentExecutionValues.Add(executionValues[t][currentExecutionNetworkI]);
                    currentNeuronActivations.Add(neuronActivations[t][currentExecutionNetworkI]);
                }

                AgroupatedNetwork cNetwork = Networks[ExecutionOrder[i]];

                cNetwork.n.GetGradients(networksOutputCostGradients[i], executionValues[i], neuronActivations[i], out List<List<double>> inputGradients);

                // Save all connections executions but if the current network is executed clear the saved list because cNetwork input is cleared
                // This can handle multiple executions of different networks without cNetwork being executed
                List<int> influentialExecutedNetworks = new List<int>();
                for (int j = 0; j < i; j++)
                {
                    if (ExecutionOrder[j] == currentExecutionNetworkI)
                    {
                        influentialExecutedNetworks.Clear();
                    }
                    else if (cNetwork.IsConnectedTo(ExecutionOrder[j]))
                    {
                        influentialExecutedNetworks.Add(ExecutionOrder[j]);
                    }
                }

                foreach (var influentialNetworkI in influentialExecutedNetworks)
                {
                    // TODO: Calculate cost gradients for respecting connections and outputs
                }
            }
        }

        #endregion


        /// <summary>
        /// 
        /// </summary>
        /// <param name="index">-1 means connected to input</param>
        /// <returns></returns>
        internal List<int> GetNetworksConnectedTo(int index)
        {
            List<int> output = new List<int>();
            for (int i = 0; i < Networks.Count; i++)
            {
                if (Networks[i].IsConnectedTo(index))
                    output.Add(i);
            }
            return output;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="networkIndex"></param>
        /// <returns>Null if connection doesn't exist</returns>
        internal Connection GetOutputConnection(int networkIndex)
        {
            foreach (var connection in OutputConnections)
                if (connection.ConnectedNetworkI == networkIndex)
                    return connection;

            return null;
        }

        internal void PassOutput(double[] networkOutput, Connection connection)
        {
            Range inputRange = connection.InputRange, outputRange = connection.ConnectedOutputRange;
            for (int networkInputI = inputRange.FromI; networkInputI < inputRange.ToI; networkInputI++)
            {
                for (int inputI = outputRange.FromI; inputI < outputRange.ToI; inputI++)
                {
                    Output[networkInputI] += networkOutput[inputI] * connection.Weights[networkInputI - inputRange.FromI][inputI - outputRange.FromI];
                }
            }
        }

        private void ClearOutput() => Output = new double[OutputLength];
    }
}

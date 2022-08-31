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
                        for (int weightI = 0; weightI < cConnection.ConnectedNetworkOutputRange.Length; weightI++)
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
                    for (int weightI = 0; weightI < OutputConnections[connectionI].ConnectedNetworkOutputRange.Length; weightI++)
                    {
                        outputConnectionsGradients[connectionI][neuronI].Add(0);
                    }
                }
            }

            // TODO: Pass output costs to output connected networks

            for (int i = ExecutionOrder.Count - 1; i >= 0; i--)
            {
                int currentExecutionIndex = ExecutionOrder[i];

                // Putting relevant values in a better formatted manner for training, a list representing time
                List<double[]> currentCostGradients = new List<double[]>();
                List < List<NeuronExecutionValues[]>> currentExecutionValues = new List<List<NeuronExecutionValues[]>>();
                List < List<double[]>> currentNeuronActivations = new List<List<double[]>>();

                for (int t = 0; t < tSCount; t++)
                {
                    currentCostGradients.Add(costGradients[t][currentExecutionIndex]);
                    currentExecutionValues.Add(executionValues[t][currentExecutionIndex]);
                    currentNeuronActivations.Add(neuronActivations[t][currentExecutionIndex]);
                }

                AgroupatedNetwork cNetwork = Networks[ExecutionOrder[i]];

                cNetwork.n.GetGradients(networksOutputCostGradients[i], executionValues[i], neuronActivations[i], out List<List<double>> inputGradients);

                // Save all connections executions but if the current network is executed clear the saved list because cNetwork input is cleared
                // This can handle multiple executions of different networks without cNetwork being executed
                List<int> influentialExecutedNetworks = new List<int>();
                for (int j = 0; j < i; j++)
                {
                    if (ExecutionOrder[j] == currentExecutionIndex)
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
                    // TODO: Calculate cost gradients for respecting connections

                    // TODO: Calculate output costs for connected networks
                }
            }
        }

        #endregion
    }
}

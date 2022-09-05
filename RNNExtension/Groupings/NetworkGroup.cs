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
        public List<AgroupatedNetwork> Networks;

        /// <summary>
        /// For proper training don't put an output connected network twice or more times
        /// </summary>
        public List<int> ExecutionOrder;
        internal List<Connection> OutputConnections;

        public readonly int InputLength;
        public readonly int OutputLength;

        public double learningRate;
        private double[] Output;

        public NetworkGroup(int inputLength, int outputLength, double learningRate)
        {
            Networks = new List<AgroupatedNetwork>();
            ExecutionOrder = new List<int>();
            OutputConnections = new List<Connection>();
            InputLength = inputLength;
            OutputLength = outputLength;
            this.learningRate = learningRate;
        }

        public double[] Execute(double[] input) => Execute(input, out _, out _, out _);

        internal double[] Execute(double[] input, out List<List<NeuronExecutionValues[]>> networksNeuronExecutionValues, out List<List<double[]>> networksNeuronOutputs, out List<double[]> groupExecutionOutputs)
        {
            ClearOutput();

            groupExecutionOutputs = new List<double[]>();

            List<int> inputConnectedNetworks = GetNetworksConnectedTo(-1);
            foreach (var networkI in inputConnectedNetworks)
            {
                Networks[networkI].GetConnectionConnectedTo(-1, out int connectionI);
                Networks[networkI].PassInput(input, connectionI, InputLength);
            }

            networksNeuronExecutionValues = new List<List<NeuronExecutionValues[]>>();
            networksNeuronOutputs = new List<List<double[]>>();

            for (int i = 0; i < ExecutionOrder.Count; i++)
            {
                int currentExecutionNetwork = ExecutionOrder[i];
                var cNetwork = Networks[currentExecutionNetwork];

                // TODO: optimize .ToArray()
                double[] nOutput = cNetwork.n.Execute(cNetwork.input.ToArray(), out List<NeuronExecutionValues[]> neuronExecutionValues, out List<double[]> neuronActivations);

                networksNeuronExecutionValues.Add(neuronExecutionValues);
                networksNeuronOutputs.Add(neuronActivations);

                List<int> networksConnectedToCurrentNetwork = GetNetworksConnectedTo(currentExecutionNetwork);
                foreach (var networkIConnectedToCurrentNetwork in networksConnectedToCurrentNetwork)
                {
                    Networks[networkIConnectedToCurrentNetwork].GetConnectionConnectedTo(currentExecutionNetwork, out int connectionI);
                    Networks[networkIConnectedToCurrentNetwork].PassInput(nOutput, connectionI, Networks[networkIConnectedToCurrentNetwork].n.OutputLength);
                }

                Connection outputConnection = GetOutputConnection(currentExecutionNetwork);
                if (outputConnection != null)
                    PassOutput(nOutput, outputConnection);

                groupExecutionOutputs.Add(Output);
            }

            return Output;
        }

        #region Gradient Learning

        public void TrainBySupervisedLearning(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction) =>
            SubtractGrads(GetSupervisedLearningGradients(X, y, costFunction));


        /// <summary>
        /// TODO: out average cost
        /// </summary>
        /// <param name="X"></param>
        /// <param name="y"></param>
        /// <param name="costFunction"></param>
        /// <returns></returns>
        internal NetworkGroupGradients GetSupervisedLearningGradients(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction)
        {
            List<double[]> costGradients = new List<double[]>();
            List<List<List<NeuronExecutionValues[]>>> executionValues = new List<List<List<NeuronExecutionValues[]>>>();
            List<List<List<double[]>>> neuronOutputs = new List<List<List<double[]>>>();
            List<List<double[]>> executionsOutputs = new List<List<double[]>>();

            DeleteMemories();
            for (int i = 0; i < X.Count; i++)
            {
                var output = Execute(X[i], out List<List<NeuronExecutionValues[]>> networksNeuronExecutionValues, out List<List<double[]>> networksNeuronOutputs, out List<double[]> groupExecutionOutputs);

                costGradients.Add(Derivatives.DerivativeOf(output, y[i], costFunction));
                executionValues.Add(networksNeuronExecutionValues);
                neuronOutputs.Add(networksNeuronOutputs);
                executionsOutputs.Add(groupExecutionOutputs);
            }
            DeleteMemories();

            return GetGradients(costGradients, executionValues, neuronOutputs, executionsOutputs);
        }

        internal NetworkGroupGradients GetGradients(List<double[]> costGradients, List<List<List<NeuronExecutionValues[]>>> executionValues, List<List<List<double[]>>> neuronActivations, List<List<double[]>> groupOutputs)
        {
            var networkGradients = GetGradients(costGradients, out List<List<List<Connection>>> connectionsGradients, out List<List<Connection>> outputConnectionsGradients, executionValues, neuronActivations, groupOutputs);
            var output = new NetworkGroupGradients()
            {
                NetworksGradients = networkGradients,
                ConnectionsGradients = connectionsGradients,
                ExecutionOrder = ExecutionOrder,
                outputConnectionsGradients = outputConnectionsGradients
            };
            return output;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="costGradients">upper list is cronologically ordered containing an array of output costs</param>
        /// <param name="connectionsGradients">list representing time, then, list ExecutionOrder execution I, containing lists representing connections gradients of the network</param>
        /// <param name="executionValues">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="neuronActivations">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="outputConnectionsGradients">List representing time, then the list of output connections gradients ordered by index</param>
        /// <param name="groupOutputs">List represting time then list of Execution order, of group outputs</param>
        /// <returns>List that represents execution order containing network gradients</returns>
        internal List<List<List<NeuronHolder>>> GetGradients(List<double[]> costGradients, out List<List<List<Connection>>> connectionsGradients, out List<List<Connection>> outputConnectionsGradients, List<List<List<NeuronExecutionValues[]>>> executionValues, List<List<List<double[]>>> neuronActivations, List<List<double[]>> groupOutputs)
        {
            List<List<List<NeuronHolder>>> output = new List<List<List<NeuronHolder>>>();
            int tSCount = groupOutputs.Count;

            // NetworkOutputGradients is a list representing execution order containing other list representing time containing a cost arrays for networks
            List<List<double[]>> networksOutputCostGradients = new List<List<double[]>>();

            // Input gradients
            networksOutputCostGradients.Add(new List<double[]>());
            for (int t = 0; t < tSCount; t++)
            {
                networksOutputCostGradients[0].Add(new double[InputLength]);
            }

            for (int i = 0; i < ExecutionOrder.Count; i++)
            {
                networksOutputCostGradients.Add(new List<double[]>());
                for (int t = 0; t < tSCount; t++)
                {
                    networksOutputCostGradients[i + 1].Add(new double[Networks[ExecutionOrder[i]].n.OutputLength]);
                }
            }

            connectionsGradients = new List<List<List<Connection>>>();
            for (int t = 0; t < tSCount; t++)
            {
                connectionsGradients.Add(new List<List<Connection>>());
                for (int executionI = 0; executionI < ExecutionOrder.Count; executionI++)
                {
                    connectionsGradients[t].Add(new List<Connection>());
                    var cNetwork = Networks[ExecutionOrder[executionI]];
                    for (int connectionI = 0; connectionI < cNetwork.Connections.Count; connectionI++)
                    {
                        connectionsGradients[t][executionI].Add(null);
                    }
                }
            }

            outputConnectionsGradients = new List<List<Connection>>();

            // Pass output costs to output connected networks
            for (int t = 0; t < tSCount; t++)
            {
                outputConnectionsGradients.Add(new List<Connection>());
                for (int i = 0; i < OutputConnections.Count; i++)
                {
                    int connectedNetworkI = OutputConnections[i].ConnectedNetworkI;
                    var connectedNetwork = Networks[connectedNetworkI].n;

                    List<int> connectedNetworkExecutionsI = new List<int>();
                    for (int j = 0; j < ExecutionOrder.Count; j++)
                        if (ExecutionOrder[j] == connectedNetworkI)
                            connectedNetworkExecutionsI.Add(j);

                    foreach (var connectedNetworkExecutionI in connectedNetworkExecutionsI)
                    {
                        List<double> connectedNetworkOutputGradients = OutputConnections[i].GetGradients(new List<double>(costGradients[t]), neuronActivations[t][connectedNetworkExecutionI][connectedNetwork.Length], out Connection weightGradients, OutputLength);
                        outputConnectionsGradients[t].Add(weightGradients);

                        var connectedNetwokOutputCost = SubtractLists(connectedNetworkOutputGradients, networksOutputCostGradients[connectedNetworkExecutionI + 1][t]);
                        //                                                  +1 counts for input
                        networksOutputCostGradients[connectedNetworkExecutionI + 1][t] = connectedNetwokOutputCost;
                    }
                }
            }

            for (int i = ExecutionOrder.Count - 1; i >= 0; i--)
            {
                int currentExecutionNetworkI = ExecutionOrder[i];

                // Putting relevant values in format for training, a list representing time containing network values
                List < List<NeuronExecutionValues[]>> currentExecutionValues = new List<List<NeuronExecutionValues[]>>();
                List < List<double[]>> currentNeuronActivations = new List<List<double[]>>();

                for (int t = 0; t < tSCount; t++)
                {
                    currentExecutionValues.Add(executionValues[t][currentExecutionNetworkI]);
                    currentNeuronActivations.Add(neuronActivations[t][currentExecutionNetworkI]);
                }

                AgroupatedNetwork cNetwork = Networks[ExecutionOrder[i]];

                // Input gradients are a list representing neurons with each neuron having a list of timesteps
                List<List<NeuronHolder>> cNetworkGradients = cNetwork.n.GetGradients(networksOutputCostGradients[i + 1], currentExecutionValues, currentNeuronActivations, out List<List<double>> inputGradients);
                output.Add(cNetworkGradients);

                // Save all connections executions but if the current network is executed clear the saved list because connectedNetwork input is cleared
                // This can handle multiple executions of different networks without connectedNetwork being executed
                List<int> influentialExecutionsIndexes = new List<int>();
                for (int j = 0; j < i; j++)
                {
                    if (ExecutionOrder[j] == currentExecutionNetworkI)
                    {
                        influentialExecutionsIndexes.Clear();
                    }
                    else if (cNetwork.IsConnectedTo(ExecutionOrder[j]))
                    {
                        influentialExecutionsIndexes.Add(j);
                    }
                }

                // Calculate cost gradients for respecting connections and networks outputs
                for (int t = 0; t < tSCount; t++)
                {
                    // Parse input gradients to a list of the current t containing the gradients of each neuron
                    var currentInputGradients = new List<double>();
                    for (int j = 0; j < inputGradients.Count; j++)
                    {
                        currentInputGradients.Add(inputGradients[j][t]);
                    }

                    foreach (var influentialExecutionI in influentialExecutionsIndexes)
                    {
                        Connection cConnection = cNetwork.GetConnectionConnectedTo(ExecutionOrder[influentialExecutionI], out int connectionI);

                        //                                                                                              there isn't a -1 because inpu layer isn't instantiated and neuron activations counts input
                        double[] connectedNetworkOutput = neuronActivations[t][ExecutionOrder[influentialExecutionI]][cNetwork.n.Length];

                        List<double> connectedNetworkOutputCosts = cConnection.GetGradients(currentInputGradients, connectedNetworkOutput, out Connection weightGradients, cNetwork.n.InputLength);

                        networksOutputCostGradients[influentialExecutionI + 1][t] = SubtractLists(connectedNetworkOutputCosts, networksOutputCostGradients[influentialExecutionI + 1][t]);
                        connectionsGradients[t][currentExecutionNetworkI][connectionI] = weightGradients;
                    }
                }

                // TODO: Calculate cost gradients for input connected connections

            }

            output.Reverse();
            return output;
        }

        internal void SubtractGrads(NetworkGroupGradients gradients)
        {
            int tSCount = gradients.outputConnectionsGradients.Count;
            for (int i = 0; i < ExecutionOrder.Count; i++)
                Networks[ExecutionOrder[i]].n.SubtractGrads(gradients.NetworksGradients[i], learningRate);

            for (int t = 0; t < tSCount; t++)
            {
                for (int i = 0; i < OutputConnections.Count; i++)
                    OutputConnections[i].SubtractGrads(gradients.outputConnectionsGradients[t][i], learningRate);

                for (int i = 0; i < ExecutionOrder.Count; i++)
                {
                    int cNetworkI = ExecutionOrder[i];
                    for (int connectionI = 0; connectionI < Networks[cNetworkI].Connections.Count; connectionI++)
                        Networks[ExecutionOrder[i]].Connections[connectionI].SubtractGrads(gradients.ConnectionsGradients[t][i][connectionI], learningRate);
                }
            }
        }

        #endregion

        /// <summary>
        /// Use it when not training because when training its done automatically
        /// </summary>
        public void DeleteMemories()
        {
            for (int i = 0; i < Networks.Count; i++)
                Networks[i].n.DeleteMemory();
        }

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
        /// Connections are backward connected, that means, input is connected to output and not otherwise, Ranges are inclusive and you can use Range.WholeRange
        /// </summary>
        /// <param name="connectedNetworkOutputRange">-1 means connected to input</param>
        public void Connect(int fromNetworkI, Range fromInputRange, int toNetworkI, Range connectedNetworkOutputRange)
        {
            if (fromNetworkI == toNetworkI || fromNetworkI < 0 || toNetworkI < -1 || fromNetworkI >= Networks.Count || toNetworkI >= Networks.Count)
                throw new ArgumentException();

            var cNetwork = Networks[fromNetworkI].n;
            Networks[fromNetworkI].Connect(toNetworkI, toNetworkI == -1? InputLength : Networks[toNetworkI].n.OutputLength, fromInputRange, connectedNetworkOutputRange, cNetwork.MaxWeight, cNetwork.MinWeight, cNetwork.WeightClosestTo0);
        }

        public void ConnectToOutput(int networkI, Range networkOutputRange, Range groupOutputRange)
        {
            var cNetwork = Networks[networkI].n;
            OutputConnections.Add(new Connection(groupOutputRange, networkI, networkOutputRange, OutputLength, cNetwork.OutputLength, cNetwork.MaxWeight, cNetwork.MinWeight, cNetwork.WeightClosestTo0));
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
            Range inputRange = connection.NetworkInputRange, outputRange = connection.ConnectedOutputRange;
            for (int networkInputI = inputRange.FromI; networkInputI < inputRange.ToI; networkInputI++)
            {
                for (int inputI = outputRange.FromI; inputI < outputRange.ToI; inputI++)
                {
                    Output[networkInputI] += networkOutput[inputI] * connection.Weights[networkInputI - inputRange.FromI][inputI - outputRange.FromI];
                }
            }
        }

        private void ClearOutput() => Output = new double[OutputLength];

        private static double[] SubtractLists(List<double> a, double[] b)
        {
            double[] output = new double[b.Length];
            for (int i = 0; i < a.Count; i++)
                output[i] = a[i] - b[i];
            return output;
        }

        public void AddN(RNN n) => Networks.Add(new AgroupatedNetwork(n));
    }
}

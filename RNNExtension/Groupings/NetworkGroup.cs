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

        /// <summary>
        /// For proper training don't put an output connected network twice or more times
        /// </summary>
        public List<int> ExecutionOrder;
        public List<Connection> OutputConnections;

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

        public double[] Execute(double[] input) => Execute(input, out _, out _, out _);

        internal double[] Execute(double[] input, out List<List<NeuronExecutionValues[]>> networksNeuronExecutionValues, out List<List<double[]>> networksNeuronOutputs, out List<double[]> groupExecutionOutputs)
        {
            ClearOutput();

            groupExecutionOutputs = new List<double[]>();

            List<int> inputConnectedNetworks = GetNetworksConnectedTo(-1);
            foreach (var networkI in inputConnectedNetworks)
            {
                Connection inputConnectedConnection = Networks[networkI].GetConnectionConnectedTo(-1);
                Networks[networkI].PassInput(input, inputConnectedConnection);
            }

            networksNeuronExecutionValues = new List<List<NeuronExecutionValues[]>>();
            networksNeuronOutputs = new List<List<double[]>>();

            for (int i = 0; i < ExecutionOrder.Count; i++)
            {
                int currentExecutionNetwork = ExecutionOrder[i];

                double[] nOutput = Networks[currentExecutionNetwork].n.Execute(input, out List<NeuronExecutionValues[]> 
                    neuronExecutionValues, out List<double[]> neuronActivations);

                networksNeuronExecutionValues.Add(neuronExecutionValues);
                networksNeuronOutputs.Add(neuronActivations);

                List<int> networksConnectedToCurrentNetwork = GetNetworksConnectedTo(currentExecutionNetwork);
                foreach (var networkIConnectedToCurrentNetwork in networksConnectedToCurrentNetwork)
                {
                    Networks[networkIConnectedToCurrentNetwork].PassInput(nOutput, Networks[networkIConnectedToCurrentNetwork].GetConnectionConnectedTo(currentExecutionNetwork));
                }

                Connection outputConnection = GetOutputConnection(currentExecutionNetwork);
                if (outputConnection != null)
                    PassOutput(nOutput, outputConnection);

                groupExecutionOutputs.Add(Output);
            }

            return Output;
        }

        #region Gradient Learning

        /// <summary>
        /// 
        /// </summary>
        /// <param name="costGradients">upper list is cronologically ordered containing an array of output costs</param>
        /// <param name="connectionsGradients"> List representing networks networkI, containing lists representing connections gradients of the network</param>
        /// <param name="executionValues">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="neuronActivations">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="outputConnectionsGradients">List representing time, then the list of output connections gradients ordered by index</param>
        /// <param name="groupOutputs">List represting time then list of Execution order, of group outputs</param>
        /// <returns></returns>
        internal List<List<List<NeuronHolder>>> GetGradients(List<double[]> costGradients, out List<List<Connection>> connectionsGradients, out List<List<Connection>> outputConnectionsGradients, List<List<List<NeuronExecutionValues[]>>> executionValues, List<List<List<double[]>>> neuronActivations, List<List<double[]>> groupOutputs)
        {
            int tSCount = groupOutputs.Count;

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

            // TODO: Pass output costs to output connected networks
            for (int t = 0; t < tSCount; t++)
            {
                for (int i = 0; i < OutputConnections.Count; i++)
                {
                    int connectedNetworkI = OutputConnections[i].ConnectedNetworkI;
                    var connectedNetwork = Networks[connectedNetworkI].n;

                    List<int> connectedNetworkExecutionsI = new List<int>();
                    for (int i = 0; i < ExecutionOrder.Count; i++)
                    {

                    }

                    foreach (var connectedNetworkExecutionI in connectedNetworkExecutionsI)
                    {
                        List<double> connectedNetworkOutputGradients = OutputConnections[i].GetGradients(new List<double>(costGradients[t]), neuronActivations[t][connectedNetworkI][connectedNetwork.Length - 1], out Connection weightGradients, connectedNetwork.InputLength);
                        outputConnectionsGradients[t].Add(weightGradients);

                        networksOutputCostGradients[connectedNetworkExecutionI][t] = AddLists(connectedNetworkOutputGradients, networksOutputCostGradients[connectedNetworkExecutionI][t]);
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

                cNetwork.n.GetGradients(networksOutputCostGradients[i], executionValues[i], neuronActivations[i], out List<List<double>> inputGradients);

                // Save all connections executions but if the current network is executed clear the saved list because connectedNetwork input is cleared
                // This can handle multiple executions of different networks without connectedNetwork being executed
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
                    // TODO: Calculate cost gradients for respecting connections and networks outputs
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

        private double[] AddLists(List<double> a, double[] b)
        {
            double[] output = new double[b.Length];
            for (int i = 0; i < a.Count; i++)
                output[i] = a[i] + b[i];
            return output;
        }
    }
}

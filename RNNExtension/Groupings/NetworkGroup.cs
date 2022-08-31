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

        public NetworkGroup()
        {

        }

        #region Gradient Learning

        /// <summary>
        /// 
        /// </summary>
        /// <param name="costGradients">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="executionValues">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <param name="neuronActivations">upper list is cronologically ordered containing list of executions in execution order</param>
        /// <returns></returns>
        public List<List<List<NeuronHolder>>> GetGradients(List<List<double[]>> costGradients, List<List<List<NeuronExecutionValues[]>>> executionValues, List<List<List<double[]>>> neuronActivations)
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
            

            for (int i = ExecutionOrder.Count - 1; i >= 0; i--)
            {
                int currentExecutionIndex = ExecutionOrder[i];

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

                cNetwork.n.GetGradients(costGradients[i], executionValues[i], neuronActivations[i], out List<List<double>> inputGradients);

                // Save all connections executions but if the current network is executed clear the saved list because cNetwork input is cleared
                // This can handle multiple executions of different networks without cNetwork being executed
                List<int> influentialExecutedNetworks = new List<int>();
                for (int j = 0; j < i; j++)
                {
                    influentialExecutedNetworks.Add(ExecutionOrder[j]);
                    if (ExecutionOrder[j] == ExecutionOrder[i])
                        influentialExecutedNetworks.Clear();
                }
            }
        }

        #endregion
    }
}

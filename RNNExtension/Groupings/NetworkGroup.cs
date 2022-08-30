using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;

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
        /// <param name="costGradients">upper list is in order of execution</param>
        /// <param name="executionValues">upper list is in order of execution</param>
        /// <param name="neuronActivations">upper list is in order of execution</param>
        /// <returns></returns>
        public List<List<List<NeuronHolder>>> GetGradients(List<List<double[]>> costGradients, List<List<List<NeuronExecutionValues[]>>> executionValues, List<List<List<double[]>>> neuronActivations)
        {
            int tCount = costGradients[0].Count;

            // NetworkOutputGradients is a list representing execution order containing other list representing time containing a cost array
            List<List<double[]>> networkOutputGradients = new List<List<double[]>>

            for (int i = ExecutionOrder.Count - 1; i >= 0; i--)
            {
                AgroupatedNetwork cNetwork = Networks[ExecutionOrder[i]];

                cNetwork.n.GetGradients(costGradients[i], executionValues[i], neuronActivations[i], out List<List<double>> inputGradients);

                foreach (var connection in cNetwork.Connections)
                {

                }
            }
        }

        #endregion
    }
}

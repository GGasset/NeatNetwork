using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Libraries;

namespace NeatNetwork.Groupings
{
    /// <summary>
    /// A connection is backward connected
    /// </summary>
    public class Connection
    {
        internal Range InputRange { get; private set; }
        internal int ConnectedNetworkI;
        internal Range OutputRange { get; private set; }

        /// <summary>
        /// List containing input neuron weights, input connected to output
        /// </summary>
        internal List<List<double>> Weights;

        internal Connection(Range inputRange, int connectedNetworkI, Range connectedNetworkOutputRange, int inputLength, int connectedNetworkOutputLength, double maxWeight, double minWeight, double weightClosestTo0)
        {
            InputRange = inputRange;
            ConnectedNetworkI = connectedNetworkI;
            OutputRange = connectedNetworkOutputRange;

            int inputRangeLength = inputRange.Length;
            inputRangeLength = Math.Min(inputRangeLength, inputLength);
            inputRangeLength += inputLength - inputRangeLength * Convert.ToInt32(inputRange == Range.WholeRange);

            int outputRangeLength = connectedNetworkOutputRange.Length;
            outputRangeLength = Math.Min(outputRangeLength, connectedNetworkOutputLength);
            outputRangeLength += outputRangeLength - connectedNetworkOutputLength * Convert.ToInt32(inputRange == Range.WholeRange);

            Weights = new List<List<double>>();
            for (int i = 0; i < inputRangeLength; i++)
            {
                Weights.Add(new List<double>());
                for (int j = 0; j < outputRangeLength; j++)
                {
                    Weights[i].Add(ValueGeneration.GenerateWeight(minWeight, maxWeight, weightClosestTo0));
                }
            }
        }

        internal void AddInputNeuron(double outputLength, bool incrementFromI, double maxWeight, double minWeight, double weightClosestTo0)
        {
            InputRange.ToI += Convert.ToInt32(incrementFromI);
            Weights.Add(new List<double>());
            int i = Weights.Count - 1;
            for (int j = 0; j < outputLength; j++)
            {
                Weights[i].Add(ValueGeneration.GenerateWeight(minWeight, maxWeight, weightClosestTo0));
            }
        }

        internal void AddOutputNeuron(bool incrementToI, double maxWeight, double minWeight, double weightClosestTo0)
        {
            OutputRange.ToI += Convert.ToInt32(incrementToI);
            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i].Add(ValueGeneration.GenerateWeight(minWeight, maxWeight, weightClosestTo0));
            }
        }
    }
}

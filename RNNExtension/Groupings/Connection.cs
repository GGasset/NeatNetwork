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
        internal Range NetworkInputRange { get; private set; }
        internal int ConnectedNetworkI;
        internal Range ConnectedOutputRange { get; private set; }

        /// <summary>
        /// List containing input neuron weights, input connected to output
        /// </summary>
        internal List<List<double>> Weights;

        internal Connection(Range inputRange, int connectedNetworkI, Range connectedNetworkOutputRange, int inputLength, int connectedNetworkOutputLength, double maxWeight, double minWeight, double weightClosestTo0)
        {
            NetworkInputRange = inputRange;
            ConnectedNetworkI = connectedNetworkI;
            ConnectedOutputRange = connectedNetworkOutputRange;

            int inputRangeLength = inputRange.Length;
            inputRangeLength = Math.Min(inputRangeLength, inputLength);
            inputRangeLength += inputLength - inputRangeLength * Convert.ToInt32(inputRange == Range.WholeRange);

            int outputRangeLength = connectedNetworkOutputRange.Length;
            outputRangeLength = Math.Min(outputRangeLength, connectedNetworkOutputLength);
            outputRangeLength += connectedNetworkOutputLength - outputRangeLength * Convert.ToInt32(inputRange == Range.WholeRange);

            Weights = new List<List<double>>();
            for (int i = 0; i < inputRangeLength; i++)
            {
                Weights.Add(new List<double>());
                for (int j = 0; j <= outputRangeLength; j++)
                {
                    Weights[i].Add(ValueGeneration.GenerateWeight(minWeight, maxWeight, weightClosestTo0));
                }
            }
        }

        #region Gradient learning

        /// <summary>
        /// 
        /// </summary>
        /// <param name="wholeInputCostGradients"></param>
        /// <returns>List of size of the connected network output</returns>
        internal List<double> GetGradients(List<double> wholeInputCostGradients, double[] connectedNetworkOutput, out Connection weightGradients, int NetworkInputLength)
        {
            List<double> output = new List<double>();
            for (int i = 0; i < connectedNetworkOutput.Length; i++)
                output.Add(0);

            weightGradients = new Connection(NetworkInputRange, ConnectedNetworkI, ConnectedOutputRange, NetworkInputLength, connectedNetworkOutput.Length, 0, 0, 0);

            Range inputRange = FormatRange(NetworkInputLength, connectedNetworkOutput.Length, true);
            Range connectedOutputRange = FormatRange(NetworkInputLength, connectedNetworkOutput.Length, false);

            for (int neuronI = inputRange.FromI; neuronI <= inputRange.ToI; neuronI++)
            {
                for (int weightI = connectedOutputRange.FromI; weightI <= connectedOutputRange.ToI; weightI++)
                {
                    output[weightI] -= wholeInputCostGradients[neuronI] * Weights[neuronI - inputRange.FromI][weightI - connectedOutputRange.FromI];
                    weightGradients.Weights[neuronI - inputRange.FromI][weightI - connectedOutputRange.FromI] += wholeInputCostGradients[neuronI] * connectedNetworkOutput[weightI];
                }
            }

            return output;
        }

        internal void SubtractGrads(Connection connectionGradients, double learningRate)
        {
            for (int neuronI = 0; neuronI < Weights.Count; neuronI++)
                for (int weightI = 0; weightI < Weights[neuronI].Count; weightI++)
                    Weights[neuronI][weightI] -= connectionGradients.Weights[neuronI][weightI] * learningRate;
        }


        #endregion

        internal void AddInputNeuron(double outputLength, double maxWeight, double minWeight, double weightClosestTo0)
        {
            Weights.Add(new List<double>());
            int i = Weights.Count - 1;
            for (int j = 0; j < outputLength; j++)
            {
                Weights[i].Add(ValueGeneration.GenerateWeight(minWeight, maxWeight, weightClosestTo0));
            }
        }

        internal void AddConnectedOutputNeuron(double maxWeight, double minWeight, double weightClosestTo0)
        {
            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i].Add(ValueGeneration.GenerateWeight(minWeight, maxWeight, weightClosestTo0));
            }
        }

        internal Range FormatRange(int NetworkInputLength, int connectedOutputLength, bool isInputRange) 
            => new Range
            (
                NetworkInputRange.FromI * Convert.ToInt32(isInputRange) + ConnectedOutputRange.FromI * Convert.ToInt32(!isInputRange)
                ,
                NetworkInputRange.ToI * Convert.ToInt32(isInputRange) + (NetworkInputLength) * Convert.ToInt32(isInputRange && NetworkInputRange == Range.WholeRange)
                +
                ConnectedOutputRange.ToI * Convert.ToInt32(!isInputRange) + (connectedOutputLength) * Convert.ToInt32(!isInputRange && ConnectedOutputRange == Range.WholeRange)
            );
    }
}

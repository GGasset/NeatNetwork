using System;
using System.Collections.Generic;
using System.Drawing;

namespace NeatNetwork.Libraries
{
    public class NeuronConnectionsInfo
    {
        internal List<Point> connectedNeuronsPos;
        internal List<double> weights;

        internal NeuronConnectionsInfo()
        {
            weights = new List<double>();
            connectedNeuronsPos = new List<Point>();
        }

        internal NeuronConnectionsInfo(List<Point> connectedNeuronsPos, List<double> weights)
        {
            this.connectedNeuronsPos = connectedNeuronsPos;
            this.weights = weights;
        }

        /// <summary>
        /// Layer 0 is input layer
        /// </summary>
        internal void AddNewConnection(int layerIndex, int neuronIndex, double minValue, double maxValue, double valueClosestTo0)
        {
            AddNewConnection(layerIndex, neuronIndex, GenerateWeight(minValue, maxValue, valueClosestTo0));
        }

        /// <summary>
        /// Layer 0 is input layer
        /// </summary>
        internal void AddNewConnection(int layerIndex, int neuronIndex, double weight)   
        {
            connectedNeuronsPos.Add(new Point(layerIndex, neuronIndex));
            weights.Add(weight);
        }

        internal void AdjustToNewLayerBeingAdded(int layerInsertionIndex, bool isinsertedInPreviousLayer, int insertedLayerLength, double minWeight, double maxWeight, double weightClosestTo0)
        {
            for (int i = 0; i < connectedNeuronsPos.Count; i++)
            {
                connectedNeuronsPos[i].Offset(Convert.ToInt32(layerInsertionIndex <= connectedNeuronsPos[i].X), 0);
            }

            if (!isinsertedInPreviousLayer)
                return;

            for (int i = 0; i < insertedLayerLength; i++)
            {
                AddNewConnection(layerInsertionIndex, i, minWeight, maxWeight, weightClosestTo0);
            }
        }


        static int randomI = int.MinValue;

        public static double GenerateWeight(double minValue, double maxValue, double valueClosestTo0)
        {
            Random r = new Random(DateTime.Now.Millisecond + randomI);
            randomI++;

            (minValue, maxValue) = (Math.Min(minValue, maxValue), Math.Max(minValue, maxValue));

            double v;
            // set is negative to -1 or 1
            int isNegative = r.Next(0, 2);
            isNegative -= Convert.ToInt32(isNegative == 0);

            //if max value is negative convert is negative to -1
            isNegative -= 2 * Convert.ToInt32(maxValue < 0);
            //if min value is positive convert is negative to 1
            isNegative += 2 * Convert.ToInt32(minValue > 0);

            valueClosestTo0 = Math.Abs(valueClosestTo0);

            // Set value closest to 0 to the closest value to 0 in respect with min/max value only if both values are positive or negative
            valueClosestTo0 += (minValue - valueClosestTo0) * Convert.ToInt32(minValue > 0);
            valueClosestTo0 -= (valueClosestTo0 - maxValue) * Convert.ToInt32(maxValue < 0);

            v = valueClosestTo0 * isNegative;
            double randomness = r.NextDouble();
            v += (randomness * (maxValue - valueClosestTo0)) * Convert.ToInt32(isNegative == 1);
            v -= (randomness * (minValue + valueClosestTo0)) * Convert.ToInt32(isNegative == -1);

            return v;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.Libraries
{
    internal static class ValueGeneration
    {
        static int randomI = int.MinValue;

        public static double GenerateWeight(double minValue, double maxValue, double weightClosestTo0)
        {
            Random r = new Random(DateTime.Now.Millisecond + randomI++);

            (minValue, maxValue) = (Math.Min(minValue, maxValue), Math.Max(minValue, maxValue));

            double v;
            // set is negative to -1 or 1
            int isPositive = r.Next(0, 2);
            isPositive -= Convert.ToInt32(isPositive == 0);

            //if max value is negative convert is negative to -1
            isPositive -= 2 * Convert.ToInt32(maxValue < 0);
            //if min value is positive convert is negative to 1
            isPositive += 2 * Convert.ToInt32(minValue >= 0);

            weightClosestTo0 = Math.Abs(weightClosestTo0);

            // Set value closest to 0 to the closest value to 0 in respect with min/max value only if both values are positive or negative
            weightClosestTo0 += (minValue - weightClosestTo0) * Convert.ToInt32(minValue >= 0);
            weightClosestTo0 -= (weightClosestTo0 - maxValue) * Convert.ToInt32(maxValue < 0);

            v = weightClosestTo0 * isPositive;
            double randomness = r.NextDouble();
            // from v which equals WeightClosestTo0 move up to max value or min value depending if its negative
            v += randomness * (maxValue - weightClosestTo0) * Convert.ToInt32(isPositive == 1);
            v += randomness * (minValue + weightClosestTo0) * Convert.ToInt32(isPositive == -1);

            return v;
        }

        public static List<double> GenerateWeights(int weigthCount, double minValue, double maxValue, double weigthClosestTo0)
        {
            var weights = new List<double>();
            for (int i = 0; i < weigthCount; i++)
                weights.Add(GenerateWeight(minValue, maxValue, weigthClosestTo0));
            return weights;
        }

        public static List<Point> GetConnectionsConnectedPosition(int connectedLayerI, int startingConnectedNeuronIndex, int outputLength)
        {
            var connections = new List<Point>();
            for (int i = 0; i < outputLength; i++)
            {
                connections.Add(new Point(connectedLayerI, startingConnectedNeuronIndex + i));
            }
            return connections;
        }

        public static double GetVariation(double minValue, double maxValue)
        {
            (minValue, maxValue) = (Math.Min(minValue, maxValue), Math.Max(minValue, maxValue));
            double output = new Random(++randomI + DateTime.Now.Millisecond).NextDouble();
            output *= maxValue - minValue;
            output += minValue;
            return output;
        }

        public static int WillMutate(double mutationChance)
        {
            Random r = new Random(++randomI + DateTime.Now.Millisecond);
            double output = r.NextDouble();
            return Convert.ToInt32(output <= mutationChance);
        }

        public static double EvolveValue(double maxVariation, double mutationChance) => GetVariation(-maxVariation, maxVariation) * WillMutate(mutationChance);

        public static List<double[]> GetNetworkCostGrid(int inputLength, int[] shape, double[] outputCosts)
        {
            List<double[]> output = new List<double[]>
            {
                new double[inputLength]
            };

            for (int i = 0; i < shape.Length; i++)
            {
                int layerLength = shape[i];
                output.Add(new double[layerLength]);
            }

            int outputLayerLength = shape[shape.Length - 1];
            for (int i = 0; i < outputLayerLength; i++)
            {
                // Corresponds to output layerMaxMutation counting with input layerMaxMutation
                output[shape.Length][i] = outputCosts[i];
            }

            return output;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="outputCosts">A list that represents time steps which contains output layers costs</param>
        /// <returns>A 3D grid in which you select layer costs then neuron costs and finally a time step</returns>
        public static List<List<List<double>>> GetTemporalNetworkCostGrid(List<double[]> outputCosts, int inputLength, int[] shape)
        {
            List<List<List<double>>> output = new List<List<List<double>>>();
            int tSCount = outputCosts.Count;
            int lastLayerI = shape.Length;

            for (int layerI = 0; layerI <= shape.Length; layerI++)
            {
                output.Add(new List<List<double>>());

                int layerLength = layerI > 0? shape[layerI - 1] : inputLength;
                for (int neuronI = 0; neuronI < layerLength; neuronI++)
                {
                    output[layerI].Add(new List<double>());
                    for (int t = 0; t < tSCount; t++)
                    {
                        output[layerI][neuronI].Add(layerI == lastLayerI? outputCosts[t][neuronI] : 0);
                    }
                }
            }
            return output;
        }
    }
}

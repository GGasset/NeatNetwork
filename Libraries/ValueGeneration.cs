using System;
using System.Collections.Generic;
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
    }
}

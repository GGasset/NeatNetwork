using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.Libraries
{
    public class Range
    {
        public int FromI;
        public int ToI;
        public int Length => ToI - FromI;

        private int HashCode;
        private static int NextHashCode = 0;

        public Range(int from, int to)
        {
            FromI = from;
            ToI = to;

            HashCode = NextHashCode;
            NextHashCode++;
        }

        public Range WholeRange => new Range(0, -1);

        public static bool operator ==(Range a, Range b) => a.FromI == b.FromI && a.ToI == b.ToI;
        public static bool operator !=(Range a, Range b) => !(a == b);

        public override int GetHashCode()
        {
            return HashCode;
        }

        public override bool Equals(object obj)
        {
            try
            {
                Range toCompare = (Range)obj;
                return toCompare == this;
            }
            catch (Exception)
            {
                return false;
            }
        }
    }
}

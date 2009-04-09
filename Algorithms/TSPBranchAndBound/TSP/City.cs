using System;
using System.Collections.Generic;
using System.Text;

namespace TSP
{
    class City
    {
        public const double INCREASE_Y = 1000;
        public const double DECREASE_Y = 1000;
        public const double INCREASE_X = 1000;
        public const double DECREASE_X = 1000;

        public City(double x, double y)
        {
            _X = x;
            _Y = y;
        }

        private double _X;
        private double _Y;

        public double X
        {
            get { return _X; }
            set { _X = value; }
        }

        public double Y
        {
            get { return _Y; }
            set { _Y = value; }
        }

        /// <summary>
        /// how much does it cost to get from this to the destination?
        /// note that this is an asymmetric cost function
        /// </summary>
        /// <param name="destination">um, the destination</param>
        /// <returns></returns>
        public double costToGetTo (City destination) 
        {
            double magnitude = Math.Sqrt(Math.Pow(this.X - destination.X, 2) + Math.Pow(this.Y - destination.Y, 2));

            magnitude *= INCREASE_X; 

            return Math.Round(magnitude);
        }


    }
}

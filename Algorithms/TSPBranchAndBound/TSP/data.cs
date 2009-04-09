using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;

namespace TSP
{
    class data
    {
        private const int DEFAULT_SIZE = 25;

        private const int CITY_ICON_SIZE = 10;

        public const double INCREASE_Y = 1.5;
        public const double DECREASE_Y = 1;
        public const double INCREASE_X = 1;
        public const double DECREASE_X = 1;

        private City[] Cities;
        private ArrayList Route;

        private Brush cityBrushStartStyle;
        private Brush cityBrushStyle;
        private Pen routePenStyle;

        private int _seed;
        private int _size;

        public int Size
        {
            get { return _size; }
        }

        public int Seed
        {
            get { return _seed; }
        }

        public data()
        {
            Random rnd = new Random();
            this._seed = rnd.Next();
            this._size = DEFAULT_SIZE;

            this.GenerateDataSet();
        }

        public data(int seed)
        {
            this._seed = seed;
            this._size = DEFAULT_SIZE;

            this.GenerateDataSet();
        }

        public data(int seed, int size)
        {
            this._seed = seed;
            this._size = size;

            this.GenerateDataSet();
        }

        private void GenerateDataSet()
        {
            Cities = new City[_size];
            Route = new ArrayList(_size);

            Random rnd = new Random(_seed);

            for (int i = 0; i < _size; i++)
                Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());

            cityBrushStyle = new SolidBrush(Color.Black);
            cityBrushStartStyle = new SolidBrush(Color.Red);
            routePenStyle = new Pen(Color.LightGray, 1);
            routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }

        public City[] GetCities()
        {
            City[] retCities = new City[Cities.Length];
            Array.Copy(Cities, retCities, Cities.Length);
            return retCities;
        }

        public void AddConnection(City city)
        {
            Route.Add(city);
        }

        public double GetCost(City from, City to)
        {
            double magnitude = Math.Sqrt(Math.Pow(from.X - to.X, 2) + Math.Pow(from.Y - to.Y, 2));

            if (to.X > from.X)
                magnitude *= INCREASE_X;
            else if (to.X < from.X)
                magnitude *= DECREASE_X;

            if (to.Y > from.Y)
                magnitude *= INCREASE_Y;
            else if (to.Y < from.Y)
                magnitude *= DECREASE_Y;

            return magnitude;
        }


        public void DrawCities(Graphics g)
        {
            float width  = g.VisibleClipBounds.Width;
            float height = g.VisibleClipBounds.Height;

            // Draw lines
            Point[] ps = new Point[Route.Count];
            int index = 0;
            foreach (City c in Route)
                ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);

            if (ps.Length > 0)
            {   
                g.DrawLines(routePenStyle, ps);
                g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width-1, (float)Cities[0].Y * height-1, CITY_ICON_SIZE+2, CITY_ICON_SIZE+2);
            }

            // Draw city dots
            foreach (City c in Cities)
            {
                g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
            }

        }

    }
}

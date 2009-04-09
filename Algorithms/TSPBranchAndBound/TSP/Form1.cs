using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Text.RegularExpressions;

namespace TSP
{
    public partial class Form1 : Form
    {
        private ProblemAndSolver CityData;
        public Form1()
        {
            InitializeComponent();

            CityData = new ProblemAndSolver();
            this.txtSeed.Text = CityData.Seed.ToString();
        }

        /// <summary>
        /// overloaded to call the redraw method for CityData. 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SetClip(new Rectangle(0,0,this.Width, this.Height - this.toolStrip1.Height-35));
            CityData.Draw(e.Graphics);
        }

        private void SetSeed()
        {
            if (Regex.IsMatch(this.txtSeed.Text, "^[0-9]+$"))
            {
                this.toolStrip1.Focus();
                CityData = new ProblemAndSolver(int.Parse(this.txtSeed.Text));
                this.Invalidate();
            }
            else
                MessageBox.Show("Seed must be an integer.");
        }

        private void Form1_Resize(object sender, EventArgs e)
        {
            this.Invalidate();
        }


        private void btnRun_Click(object sender, EventArgs e)
        {
            if (CityData.Seed.ToString() != this.txtSeed.Text)
                this.SetSeed();
            CityData.solveProblem();
        }

        private void txtSeed_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
                this.SetSeed();
        }

        private void toolStrip1_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {

        }

        private void bNewProblem_Click(object sender, EventArgs e)
        {
            if (Regex.IsMatch(this.tbProblemSize.Text, "^[0-9]+$"))
            {
                CityData.GenerateProblem(int.Parse(this.tbProblemSize.Text));
                this.Invalidate(); 
            }
            else
            {
                MessageBox.Show("Problem size must be an integer.");
            }; 
        }

    }
}
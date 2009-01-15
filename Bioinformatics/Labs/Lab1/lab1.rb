#!/usr/bin/env ruby
require 'rubygems'
require 'bio'

ff = Bio::FlatFile.new(Bio::FastaFormat, open("sequences-1.fasta.out"))
# ff.each_entry do |f|
#   puts "definition : " + f.definition
#   puts "nalen      : " + f.nalen.to_s
#   puts "naseq      : " + f.naseq
# end

seqs = ff.map{ |e| e.naseq }
# seqs = ff.map{ |e| e.naseq }
# p seqs
ff.each_entry do |a|
	puts "hi"
end

# align = Bio::Alignment.new(seqs)
# p align

# a1 = Bio::Alignment.new(["aaaact", "ctggggg"])
# a2 = Bio::Alignment.new([""])
# p a1.consensus_string
# p a2

# p a1.consensus


# s1 = Bio::Sequence::NA.new("aaactg")
# p s1.reverse

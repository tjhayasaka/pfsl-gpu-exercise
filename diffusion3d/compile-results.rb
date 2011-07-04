#!/usr/bin/env ruby

# SCORE = (A)/30 + (B)/40 + (C)/50
result = {}
nxs = [64, 128, 256]
score_weight = { 64 => 1.0 / 30, 128 => 1.0 / 40, 256 => 1.0 / 50 }

# read results

File.readlines("bench-result.txt").map { |line| line.sub(/#.*$/, "").strip }.reject(&:empty?).each do |line|
  cols = line.split
  impl = cols[0].sub(%r(^.*/), "").sub(%r(\.exe$), "").sub(/^float@/, "")
  nx = cols[1].to_i
  val = cols[2..-1].map(&:to_f)
  val = { :time => val[0], :mflops => val[1], :error => val[2] }
  result[impl] ||= {}
  result[impl][nx] = val if result[impl][nx].nil? || result[impl][nx][:mflops] < val[:mflops]
end

# calculate score

result.each do |impl, values|
  score = 0.0
  nxs.each do |nx|
    gflops = values[nx][:mflops] * 0.001
    score += gflops * score_weight[nx]
  end
  values[:score] = score
end

# reorder by score

result = result.to_a.sort_by { |v| v[1][:score] }.reverse

# print

result.each do |impl, values|
  print "#{impl}"
  nxs.each do |nx|
    value = values[nx]
    gflops = value[:mflops].to_i * 0.001
    printf("\t%7.3f", gflops)
  end
  printf("\t%7.3f\n", values[:score])
end

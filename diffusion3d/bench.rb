#!/usr/bin/env ruby

def append_result(line)
  result_filename = "bench-result.txt"
  File.open(result_filename, "a") { |f| f.puts(line) }
end

def bench_one(bin_file)
  nxs = [64, 128, 256]
  nxs.each do |nx|
    command_line = "sync;sync;sync; #{bin_file} --nx #{nx}"
    puts command_line
    GC.start
    lines = IO.popen(command_line, "r") { |f| f.readlines }.map { |line| line.sub(/.*=/, "").sub(/\[.*/, "").strip }
    record = ([bin_file, nx] + lines).join("\t")
    puts record
    append_result(record)
  end
end

def bench(bin_files)
  time_begin = Time.now
  append_result("########")
  append_result("# benchmarking start: #{Time.now}")
  append_result("# bin_files = #{bin_files.inspect}")
  append_result("# uptime = #{IO.popen("/usr/bin/uptime").read.strip}")

  bin_files.each { |bin_file| bench_one(bin_file) }

  time_end = Time.now
  append_result("# benchmarking finished: #{Time.now}, elapsed = #{time_end - time_begin}")
  append_result("########")
end

if ARGV.empty?
  bench Dir.glob("bin/float@gpu@*").shuffle
else
  bench ARGV
end

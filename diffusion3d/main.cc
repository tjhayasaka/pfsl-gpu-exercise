//

#include "IterationCounter.h"
#include "MySolver.h"
#include "ArrayAccessor.h"
#include <boost/scoped_ptr.hpp>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <fstream>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

static void usage(FILE *f, char const *progname)
{
  fprintf(f, "usage: %s [--nx grid-size] [--dump-interval=N] [--dump-filename-pattern=PATTERN]\n", progname);
}

template <typename Float>
static void initializeField(Solver<Float> const &solver, Float *data)
{
  Dim3 numNodesInGrid = solver.numNodesInGridIncludingHalo();
  ArrayAccessor3<Float> f(numNodesInGrid, data);
  int halo = solver.halo();
  Vec3<Float> cellSpacing = solver.cellSpacing();
  Vec3<Float> k = Vec3<Float>(2.0, 2.0, 2.0) * Float(M_PI);

  for (uint jz = 0; jz < numNodesInGrid.z; jz++) {
    for (uint jy = 0; jy < numNodesInGrid.y; jy++) {
      for (uint jx = 0; jx < numNodesInGrid.x; jx++) {
	Vec3<Float> x((int(jx) - halo) + Float(0.5), (int(jy) - halo) + Float(0.5), (int(jz) - halo) + Float(0.5));
	Vec3<Float> phase = k * cellSpacing * x;
	f(jx, jy, jz) = Float(0.125) * (Float(1.0) - cos(phase.x)) * (Float(1.0) - cos(phase.y)) * (Float(1.0) - cos(phase.z));
      }
    }
  }
}

template <typename Float>
static void calcurateAccuracy(Solver<Float> const &solver, Float const *data, Float kappa, Float time, Float *ret)
{
  Dim3 numNodesInGrid = solver.numNodesInGridIncludingHalo();
  ArrayAccessor3<const Float> f(numNodesInGrid, data);
  int halo = solver.halo();
  Vec3<Float> cellSpacing = solver.cellSpacing();
  Vec3<Float> k = Vec3<Float>(2.0, 2.0, 2.0) * Float(M_PI);
  Vec3<Float> ktkk = (-kappa * time) * (k * k);
  Vec3<Float> a(exp(ktkk.x), exp(ktkk.y), exp(ktkk.z));
  Float ferr = 0.0;

  for (uint jz = halo; jz < numNodesInGrid.z - halo; jz++) {
    for (uint jy = halo; jy < numNodesInGrid.y - halo; jy++) {
      for (uint jx = halo; jx < numNodesInGrid.x - halo; jx++) {
	Vec3<Float> x((jx - halo) + Float(0.5), (jy - halo) + Float(0.5), (jz - halo) + Float(0.5));
	Vec3<Float> phase = k * cellSpacing * x;
	Float f0 = Float(0.125) * (Float(1.0) - a.x * cos(phase.x)) * (Float(1.0) - a.y * cos(phase.y)) * (Float(1.0) - a.z * cos(phase.z));
	Float diff = f(jx, jy, jz) - f0;
	ferr += diff * diff;
      }
    }
  }

  *ret = sqrt(ferr / Float(solver.numTotalNodesInGrid()));
}

template <typename Float>
static void printStatus_(int currentIteration, Float const &time)
{
  // wrap fprintf to make it boost::bind'able
  fprintf(stderr, "time(%4d)=%7.5f\n", currentIteration, time);
}

template <typename Float>
class Dumper
{
public:
  Dumper() {}

  ~Dumper() {}

  void dump(Solver<Float> &solver, char const *filenamePattern, int iteration)
  {
    if (!filenamePattern || *filenamePattern == '\0')
      return;

    char filename[1024];
    snprintf(filename, sizeof(filename), filenamePattern, iteration); // BUG: FIXME: THIS IS REALLY A BIG SECURITY HOLE
    fprintf(stderr, "dump:  filename='%s'\n", filename);

    std::ofstream of(filename);
    solver.withFrontBuffer(Solver<Float>::bom_ro, boost::bind(&Dumper<Float>::dump_aux, this, boost::ref(of), _1, _2));
  }

private:
  void dump_aux(std::ofstream &of, Solver<Float> const &solver, Float const *data)
  {
    int halo = solver.halo();
    Dim3 numNodesInGrid = solver.numNodesInGridIncludingHalo();
    ArrayAccessor3<const Float> f(numNodesInGrid, data);
    of << std::fixed;
    of << "{\n"
       << "  :halo => " << halo << ",\n"
       << "  :numNodesInGridIncludingHalo => [" << numNodesInGrid.x << ", " << numNodesInGrid.y << ", " << numNodesInGrid.z << "],\n"
       << "  :data => [\n";
    for (uint jz = 0 ; jz < numNodesInGrid.z; jz++) {
      of << "\n";
      of << "# jz = " << jz << "\n";
      for (uint jy = 0 ; jy < numNodesInGrid.y; jy++) {
	of << "   ";
	for (uint jx = 0 ; jx < numNodesInGrid.x; jx++) {
	  of << " " << f(jx, jy, jz) << ",";
	}
	of << "\n";
      }
    }
    of << "  ]\n"
       << "}\n";
  }
};

template <typename Float>
static int main_(int argc, char *argv[])
{
  Dim3 numNodesInGrid(64, 0, 0);
  int opt_dump_interval = -1;
  char const *opt_dump_filenamePattern = NULL;

  for (;;) {
    static struct option long_options[] = {
      {"help",  no_argument, 0, 'h'},
      {"nx",  required_argument, 0, 'x'},
      {"ny",  required_argument, 0, 'y'},
      {"nz",  required_argument, 0, 'z'},
      {"dump-interval",  required_argument, 0, 'd'},
      {"dump-filename-pattern",  required_argument, 0, 'D'},
      {0, 0, 0, 0}
    };

    int option_index = 0;

    int c = getopt_long(argc, argv, "h", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 'h': usage(stdout, argv[0]); exit(0); break;
    case 'x': numNodesInGrid.x = atoi(optarg); break;
    case 'y': numNodesInGrid.y = atoi(optarg); break;
    case 'z': numNodesInGrid.z = atoi(optarg); break;
    case 'd': opt_dump_interval = atoi(optarg); break;
    case 'D': opt_dump_filenamePattern = optarg; break;
    case '?':
      /* `getopt_long' already printed an error message. */
      usage(stderr, argv[0]);
      exit(1);
      break;
    default:
      abort();
    }
  }

  if (optind < argc) {
    usage(stderr, argv[0]);
    exit(1);
  }

  numNodesInGrid.y = (numNodesInGrid.y <= 0 ? numNodesInGrid.x : numNodesInGrid.y);
  numNodesInGrid.z = (numNodesInGrid.z <= 0 ? numNodesInGrid.x : numNodesInGrid.z);

  if (numNodesInGrid.x != 64 && numNodesInGrid.x != 128 && numNodesInGrid.x != 256) {
    fprintf(stderr, "grid size must be 64, 128 or 256.\n");
    usage(stderr, argv[0]);
    exit(1);
  }

  //

  boost::scoped_ptr<Solver<Float> > solver(new MySolver<Float>);
  Vec3<Float> const fieldSize(1.0, 1.0, 1.0);
  Vec3<Float> const cellSpacing = fieldSize / Vec3<Float>(numNodesInGrid);
  solver->reset(numNodesInGrid, cellSpacing);
  solver->withFrontBuffer(Solver<Float>::bom_wo, &initializeField<Float>);

  Float const kappa = 0.1;
  Float const dt = Float(0.1) * cellSpacing.x * cellSpacing.x / kappa;

  Float time = 0.0;
  Float flops = .0;

  IterationCounter iteration;

  PeriodicRunner periodicStatusPrinter(100, boost::bind(&printStatus_<Float>, _1, boost::cref(time)));
  iteration.addPeriodicRunner(periodicStatusPrinter);

  Dumper<Float> dumper;
  PeriodicRunner periodicDumper(opt_dump_interval, boost::bind(&Dumper<Float>::dump, dumper, boost::ref(*solver.get()), opt_dump_filenamePattern, _1));
  iteration.addPeriodicRunner(periodicDumper);

  unsigned int realtimeTimer;
  cutCreateTimer(&realtimeTimer);
  cutResetTimer(realtimeTimer);
  cutStartTimer(realtimeTimer);

  while (iteration.currentIteration() < 90000 && time + Float(0.5) * dt < Float(0.1)) {
    iteration.nextTick();

    flops += solver->nextTick(kappa, dt);
    solver->swapBuffers();

    time += dt;
  }

  cutStopTimer(realtimeTimer);
  Float elapsedTime = cutGetTimerValue(realtimeTimer) * Float(1.0e-03);

  iteration.nextTick(true); // for dump and status

  Float accuracy_ret;
  boost::function<void (Solver<Float> const &solver, Float const *buffer)> accuracy_func = boost::bind(&calcurateAccuracy<Float>, _1, _2, kappa, time, &accuracy_ret);
  solver->withFrontBuffer(Solver<Float>::bom_ro, accuracy_func);

  printf("Elapsed Time= %9.3e [sec]\n", elapsedTime);
  printf("Performance= %7.2f [MFlops]\n", flops / elapsedTime * Float(1.0e-06));
  printf("Error[%d][%d][%d]=%10.4e\n",
	 numNodesInGrid.x, numNodesInGrid.y, numNodesInGrid.z,
	 accuracy_ret);

  return 0;
}

int main(int argc, char *argv[])
{
  main_<FLOAT>(argc, argv);
}

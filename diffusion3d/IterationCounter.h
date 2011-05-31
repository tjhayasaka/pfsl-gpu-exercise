#ifndef IterationCounter_h__
#define IterationCounter_h__ 1

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <vector>

class PeriodicRunner
{
public:
  PeriodicRunner(int interval, boost::function<void (int currentIteration)> const &func) :
    interval_(interval),
    func_(func)
  {
  }

  void nextTick(int currentIteration, bool forceRun)
  {
    bool run = false;
    run = run || (interval_ >= 0 && forceRun);
    run = run || (interval_ > 0 && currentIteration % interval_ == 0);
    if (run)
      func_(currentIteration);
  }

private:
  int interval_;
  boost::function<void (int currentIteration)> func_;
};

class IterationCounter
{
public:
  typedef std::vector<PeriodicRunner> PeriodicRunners;

  IterationCounter(int initialIteration = 0) :
    currentIteration_(initialIteration)
  {
  }

  void addPeriodicRunner(PeriodicRunner const &runner)
  {
    periodicRunners_.push_back(runner);
  }

  int currentIteration() const
  {
    return currentIteration_;
  }

  void nextTick(bool forceRun = false)
  {
    for (PeriodicRunners::iterator i = periodicRunners_.begin(); i != periodicRunners_.end(); ++i)
      i->nextTick(currentIteration_, forceRun);
    currentIteration_++;
  }

private:
  int currentIteration_;
  PeriodicRunners periodicRunners_;
};

#endif

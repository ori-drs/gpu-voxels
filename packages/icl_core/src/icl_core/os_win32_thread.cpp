// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-03-29
 *
 */
//----------------------------------------------------------------------
#include "icl_core/os_win32_thread.h"

namespace icl_core {
namespace os {
namespace hidden_win32 {

ThreadId threadSelf()
{
  return ::GetCurrentThreadId();
}

pid_t getpid()
{
  return ::_getpid();
}

}
}
}

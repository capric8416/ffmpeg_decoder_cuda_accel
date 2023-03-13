#pragma once

// c
#include <stdarg.h>
#include <stdio.h>
// c++
#include <ctime>



/******************************************************************************/
// get process, thread id
#if defined(WIN32)
#include <Windows.h>
#define get_pid (int)GetCurrentProcessId
#define get_tid (int)GetCurrentThreadId
#else
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#define get_pid (int)getpid
#define get_tid (int)pthread_self
#endif

// get current date time
#define GET_DATE_TIME(datetime, size)                               \
    std::time_t current_dt = std::time(0);                          \
    std::tm *local_time = std::localtime(&current_dt);              \
    std::strftime(datetime, size, "%Y-%m-%d %H:%M:%S", local_time);

// log formated message
#define LOG_PANIC(fmt, ...)    {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [PANIC] [%s #%d %d %d] "     fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
#define LOG_FATAL(fmt, ...)    {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [FATAL] [%s #%d %d %d] "     fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
#define LOG_ERROR(fmt, ...)    {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [ERROR] [%s #%d %d %d] "     fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
#define LOG_WARNING(fmt, ...)  {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [WARNNING] [%s #%d %d %d] "  fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
#define LOG_INFO(fmt, ...)     {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [INFO] [%s #%d %d %d] "      fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
#define LOG_VERBOSE(fmt, ...)  {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [VERBOSE] [%s #%d %d %d] "   fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
#define LOG_DEBUG(fmt, ...)    {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [DEBUG] [%s #%d %d %d] "     fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
#define LOG_TRACE(fmt, ...)    {char dt[40]; GET_DATE_TIME(dt, 40); printf("[%s] [TRACE] [%s #%d %d %d] "     fmt, dt, __FUNCTION__, __LINE__, get_pid(), get_tid(), ##__VA_ARGS__);}
/******************************************************************************/


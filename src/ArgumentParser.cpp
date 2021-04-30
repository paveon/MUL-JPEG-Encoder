#include "ArgumentParser.h"
#include <algorithm>
#include <sstream>

using namespace std;

namespace Parser {
   void ArgumentParser::ValidateArgument(const Argument& newArg) {
      if (newArg.name.empty()) {
         throw ArgException("Argument name cannot be empty");
      }
      else if (newArg.isSwitch && newArg.name[0] != '-') {
         throw ArgException("Name of switch argument must start with '-' character");
      }
      else if (!newArg.isSwitch && newArg.name[0] == '-') {
         throw ArgException("Name of positional argument cannot start with '-' character");
      }

      if (newArg.exclusive) {
         /* Check for conflicting exclusive configurations i.e. <required | optional> */
         for (const Argument& arg : m_ExpectedArgs) {
            if (arg.exclusive && (arg.optional != newArg.optional)) {
               throw ArgException("Mutually exclusive group cannot contain "
                                  "both optional and required arguments");
            }
         }
      }
      m_ExpectedArgs.push_back(newArg);
   }


   void ArgumentParser::Parse() {
      /* Main parsing loop */
      while (!m_InputArgs.empty()) {
         /* Get next argument in line */
         std::string inputArg(std::move(m_InputArgs.front()));
         m_InputArgs.pop_front();
         bool isSwitch = (inputArg[0] == '-');

         /* Check if argument is expected, use lambda predicate to find value in vector */
         auto argIt = std::find_if(m_ExpectedArgs.begin(), m_ExpectedArgs.end(), [&](const Argument& arg) {
            /* Names are compared only if expected arg is a switch, names of positional
             * arguments are used only for their retrieval after successful parsing */
            if (arg.isSwitch) {
               if (arg.name == inputArg) {
                  if (arg.alreadyParsed) {
                     throw ArgException(std::string("Duplicate '" + arg.name + "' option"));
                  }
                  else if (arg.hasValue && m_InputArgs.empty()) {
                     std::ostringstream errMsg;
                     errMsg << "Missing value for switch '" << arg.name << "'";
                     throw ArgException(errMsg.str());
                  }
                  return true;
               }
               return false;
            }

            /* Expected argument is not switch so check if input argument is.
             * If input arg is not switch then it must be positional argument and in that case
             * we don't compare anything and just check if expected arg was already parsed */
            return (!isSwitch && !arg.alreadyParsed);
         });

         if (argIt == m_ExpectedArgs.end()) {
            throw ArgException(std::string("Unknown argument '" + inputArg + "'"));
         }

         /* Check if there are exclusive collisions */
         if (argIt->exclusive) {
            if (!m_ExclusiveGroup.empty()) {
               std::ostringstream errMsg;
               errMsg << "Options '" << argIt->name << "' and '"
                      << m_ExclusiveGroup.front()->name << "' are mutually exclusive";
               throw ArgException(errMsg.str());
            }
            else {
               m_ExclusiveGroup.push_back(&(*argIt));
            }
         }

         /* Add new mapping [argument name -> value] */
         if (argIt->isSwitch) {
            if (argIt->hasValue) {
               /* Switch option with value, pop next value from input arguments */

               /* Check if value is valid */
               std::string& value = m_InputArgs.front();
               if (value[0] == '-') {
                  /* Might be an existing switch instead of value */
                  for (const Argument& arg : m_ExpectedArgs) {
                     if (value == arg.name) {
                        std::ostringstream errMsg;
                        errMsg << "Expected value for switch '" << argIt->name
                               << "', got switch '" << arg.name << "' instead";
                        throw ArgException(errMsg.str());
                     }
                  }
               }

               m_ParsedArgs[argIt->name] = Variant(std::move(value));
               m_InputArgs.pop_front();
            }
            else {
               /* Switch without value */
               m_ParsedArgs[argIt->name] = Variant(true);
            }
         }
         else {
            /* Positional argument, use current input argument as value */
            m_ParsedArgs[argIt->name] = Variant(inputArg);
         }
         argIt->alreadyParsed = true;
      }

      /* Final check if required arguments were provided */
      for (auto arg : m_ExpectedArgs) {
         if (!arg.optional && !arg.alreadyParsed) {
            /* Possibly missing argument */

            if ((arg.exclusive && m_ExclusiveGroup.empty())) {
               /* Missing argument was from exclusive group, construct error message */

               std::ostringstream errMsg;
               errMsg << "Please provide at least one required argument from <";
               for (const Argument& expected: m_ExpectedArgs) {
                  if (expected.exclusive) {
                     errMsg << expected.name;
                     if (expected.isSwitch && expected.hasValue) {
                        errMsg << " <value>";
                     }
                     errMsg << " | ";
                  }
               }
               /* Stream magic to remove the last separator */
               errMsg.seekp(-3, std::ios_base::end);
               errMsg << "> mutually exclusive group";
               throw ArgException(errMsg.str());
            }
            else if (!arg.exclusive) {
               /* Required non-exclusive argument is missing */
               throw ArgException(std::string("Missing required '" + arg.name + "' argument"));
            }
         }
      }
   }
}
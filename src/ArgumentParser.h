#ifndef ARGUMENTPARSER_H
#define ARGUMENTPARSER_H

#include <iostream>
#include <vector>
#include <list>
#include <map>


namespace Parser {
   namespace Arg {
      /* Using flags instead of boolean values to make
       * configuration of new arguments more readable */
      enum Options {
         Optional = 0x01,
         Exclusive = 0x02,
         HasValue = 0x04
      };
   }

   /* Exception class for argument parser */
   class ArgException : public std::runtime_error {
   public:
      explicit ArgException(const std::string& error) : runtime_error(error) {}
   };


   /* Helper class that can store multiple types */
   class Variant {
   private:
      enum class Type {
         EMPTY,
         BOOL,
         STRING
      } m_CurrentType;

      /* Did not use union because it doesn't allow to store C++ objects
       * but it doesn't really matter in this case...
       * I would need to use raw pointers and create more robust solution
       * with manual memory management if I decided to extend it and use
       * it in next projects with more possible types to store */
      bool valueBool;
      std::string valueString;

   public:
      Variant() : m_CurrentType(Type::EMPTY), valueBool(false), valueString() {}

      explicit Variant(bool value) : valueBool(value) {
         m_CurrentType = Type::BOOL;
      }

      explicit Variant(std::string value) : valueBool(false), valueString(std::move(value)) {
         m_CurrentType = Type::STRING;
      }

      explicit operator bool() const {
         if (m_CurrentType != Type::BOOL) {
            throw std::runtime_error("Illegal variant conversion");
         }
         return valueBool;
      }

      explicit operator const std::string&() const {
         if (m_CurrentType != Type::STRING) {
            throw std::runtime_error("Illegal variant conversion");
         }
         return valueString;
      }
   };


   /* Main parsing class */
   class ArgumentParser {
   public:

   private:
      struct Argument {
         std::string name; /* Full name of argument */
         bool hasValue; /* For switches, indicates that switch has value */
         bool optional; /* Optional / required argument */
         bool exclusive; /* Exclusive argument, only one exclusive group is currently supported */
         bool isSwitch; /* Switch / positional argument */
         bool alreadyParsed; /* State variable */
      };

      std::vector<Argument*> m_ExclusiveGroup; /* Single exclusive group */
      std::vector<Argument> m_ExpectedArgs; /* Set of configured arguments */
      std::list<std::string> m_InputArgs; /* All input arguments */
      std::map<std::string, Variant> m_ParsedArgs; /* Maps parsed arguments to their respective values */

      /* Internal method for basic validation of configured argument */
      void ValidateArgument(const Argument& newArg);


   public:
      ArgumentParser(int argc, const char* const argv[]) {
         for (int i = 1; i < argc; i++) {
            std::string arg(argv[i]);
            m_InputArgs.emplace_back(arg);
         }
      }

      /* Adds new positional argument to the parsing framework */
      void AddArgument(const std::string& name, uint8_t flags = 0) {
         bool opt = flags & Arg::Optional;
         bool exc = flags & Arg::Exclusive;
         Argument arg = {
                 name, true, opt,
                 exc, false, false
         };
         ValidateArgument(arg);
      }

      /* Adds new switch to the parsing framework */
      void AddSwitch(const std::string& name, uint8_t flags = 0) {
         bool opt = flags & Arg::Optional;
         bool exc = flags & Arg::Exclusive;
         bool val = flags & Arg::HasValue;
         Argument arg = {
                 name, val, opt,
                 exc, true, false
         };
         ValidateArgument(arg);
      }

      /* Performs parsing input arguments based on configuration */
      void Parse();

      /* Performs lookup of parsed argument values, returns
       * nullptr in case of missing optional argument */
      const Variant* operator[](const std::string& argName) const {
         /* Caching */
         static std::string lastLookup;
         static const Variant* cachedVariant(nullptr);
         if (lastLookup == argName) {
            return cachedVariant;
         }

         lastLookup = argName;
         auto it = m_ParsedArgs.find(argName);
         if (it != m_ParsedArgs.end()) {
            return (cachedVariant = &it->second);
         }
         return (cachedVariant = nullptr);
      }

      /* Wrapper for easier lookup of valueless switches */
      bool HasArgument(const std::string& argName) const {
         return (this->operator[](argName) != nullptr);
      }
   };
}


#endif //ARGUMENTPARSER_H

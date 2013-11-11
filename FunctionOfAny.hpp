#pragma once
#include <boost/unordered_map.hpp>

template< class Domain, class Codomain >
class FunctionOfAny
{
public:
	//
	typedef boost::unordered_map< Domain, Codomain > Container;
	typedef typename Container::iterator Iterator;
	typedef typename Container::const_iterator ConstIterator;
	
	// ctor / dtor
					FunctionOfAny();
					FunctionOfAny( const FunctionOfAny& functionOfAny );
					~FunctionOfAny();

	Iterator 		begin();
	Iterator 		end();

	ConstIterator 	begin() const;
	ConstIterator 	end() const;
				
	size_t 			size() const;

private:

	Container 		m_table;

};


template< class Domain, class Codomain >
FunctionOfAny< Domain, Codomain >::FunctionOfAny()
{

}


//
template< class Domain, class Codomain >
FunctionOfAny< Domain, Codomain >::FunctionOfAny( const FunctionOfAny< Domain, Codomain >& functionOfAny )
{

}


//
template< class Domain, class Codomain >
FunctionOfAny< Domain, Codomain >::~FunctionOfAny()
{

}


//
template< class Domain, class Codomain >
typename FunctionOfAny< Domain, Codomain >::Iterator FunctionOfAny< Domain, Codomain >::begin()
{
	return m_table.begin();
}


//
template< class Domain, class Codomain >
typename FunctionOfAny< Domain, Codomain >::Iterator FunctionOfAny< Domain, Codomain >::end()
{
	return m_table.end();
}


//
template< class Domain, class Codomain >
typename FunctionOfAny< Domain, Codomain >::ConstIterator FunctionOfAny< Domain, Codomain >::begin() const
{
	return m_table.begin();
}


//
template< class Domain, class Codomain >
typename FunctionOfAny< Domain, Codomain >::ConstIterator FunctionOfAny< Domain, Codomain >::end() const
{
	return m_table.end();
}


//
template< class Domain, class Codomain >
size_t FunctionOfAny< Domain, Codomain >::size() const
{
	return m_table.size();
}
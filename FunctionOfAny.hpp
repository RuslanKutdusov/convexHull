#pragma once
#include <boost/unordered_map.hpp>

template< class Domain, class Codomain >
class FunctionOfAny
{
public:
	//
	typedef boost::unordered_map< Domain, Codomain > Container;
	typedef typename Container::iterator iterator;
	typedef typename Container::const_iterator const_iterator;
	
	// ctor / dtor
					FunctionOfAny();
					FunctionOfAny( const FunctionOfAny& functionOfAny );
					~FunctionOfAny();

	iterator 		begin();
	iterator 		end();

	const_iterator 	begin() const;
	const_iterator 	end() const;
				
	size_t 			size() const;

	Codomain&		define( const Domain& x );

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
typename FunctionOfAny< Domain, Codomain >::iterator FunctionOfAny< Domain, Codomain >::begin()
{
	return m_table.begin();
}


//
template< class Domain, class Codomain >
typename FunctionOfAny< Domain, Codomain >::iterator FunctionOfAny< Domain, Codomain >::end()
{
	return m_table.end();
}


//
template< class Domain, class Codomain >
typename FunctionOfAny< Domain, Codomain >::const_iterator FunctionOfAny< Domain, Codomain >::begin() const
{
	return m_table.begin();
}


//
template< class Domain, class Codomain >
typename FunctionOfAny< Domain, Codomain >::const_iterator FunctionOfAny< Domain, Codomain >::end() const
{
	return m_table.end();
}


//
template< class Domain, class Codomain >
size_t FunctionOfAny< Domain, Codomain >::size() const
{
	return m_table.size();
}


//
template< class Domain, class Codomain >
Codomain& FunctionOfAny< Domain, Codomain >::define( const Domain& x )
{
	iterator it = m_table.find( x );
	if ( it != m_table.end() ) 
	{
		return it->second;
	} 
	else 
	{
		m_table[ x ] = Codomain();
		return m_table[ x ];
	}
}

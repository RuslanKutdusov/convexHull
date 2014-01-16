#pragma once
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>

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

	const Codomain& at( const Domain& x );

	void clear()
	{
		m_table.clear();
	}

private:

	Container 		m_table;

	//
	friend class boost::serialization::access;

	//
	template<class Archive>
	void load(Archive& ar, const unsigned int version);
	template<class Archive>
	void save(Archive& ar, const unsigned int version) const;

	BOOST_SERIALIZATION_SPLIT_MEMBER()	
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


//
template< class Domain, class Codomain >
const Codomain& FunctionOfAny< Domain, Codomain >::at( const Domain& x )
{
	return m_table[ x ];
}


//
template<class Domain, class Codomain>
template<class Archive>
void FunctionOfAny<Domain, Codomain>::save( Archive& ar, const unsigned int /* version */) const
{
    size_t size = m_table.size();
    ar << boost::serialization::make_nvp( "size", size );
    typedef typename Container::value_type value_type;
    BOOST_FOREACH( value_type p, m_table )
        ar << p;
}


//
template<class Domain, class Codomain>
template<class Archive>
void FunctionOfAny<Domain, Codomain>::load( Archive& ar, const unsigned int /* version */ )
{
	size_t size;
	typename Container::value_type p;
	ar >> boost::serialization::make_nvp( "size", size );
	m_table = Container( size );
	for( size_t i = 0; i < size; ++i ) 
	{
		ar >> p;
		m_table.insert(p);
	}
}